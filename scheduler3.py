import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import pulp as pl

DEFAULT_DOCTORS = {
    'Anderson':  {'patients': 24, 'MOKC': 0, 'SH': 1, 'MWC': 0, 'type': 'FT'},
    'Baroi':     {'patients': 20, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Blair':     {'patients': 16, 'MOKC': 0, 'SH': 1, 'MWC': 1, 'type': 'PT'},
    'Brazowski': {'patients': 24, 'MOKC': 0, 'SH': 1, 'MWC': 0, 'type': 'FT'},
    'Fogle':     {'patients': 10, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Gopala':    {'patients': 24, 'MOKC': 1, 'SH': 0, 'MWC': 1, 'type': 'FT'},
    'Langer':    {'patients': 18, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Langmacher': {'patients': 18, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Mani':      {'patients': 20, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Martin':    {'patients': 24, 'MOKC': 0, 'SH': 0, 'MWC': 1, 'type': 'FT'},
    'Miles':     {'patients': 24, 'MOKC': 0, 'SH': 0, 'MWC': 1, 'type': 'FT'},
    'Nazir':     {'patients': 24, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    "O'Neal":    {'patients': 20, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Patel':     {'patients': 24, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Pham':      {'patients': 20, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Porter':    {'patients': 20, 'MOKC': 0, 'SH': 1, 'MWC': 0, 'type': 'PT7'},
    'Riggs':     {'patients': 10, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'PTWD'},
    'Russell':   {'patients': 24, 'MOKC': 1, 'SH': 1, 'MWC': 1, 'type': 'FT'},
    'Shipman':   {'patients': 24, 'MOKC': 0, 'SH': 0, 'MWC': 1, 'type': 'PTWE'},
    'Sidney Le': {'patients': 24, 'MOKC': 0, 'SH': 0, 'MWC': 1, 'type': 'PT4'},
    'Smith':     {'patients': 20, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'FT'},
    'Treadwell': {'patients': 10, 'MOKC': 1, 'SH': 0, 'MWC': 0, 'type': 'PTWD'}
}

DEFAULT_DEMAND = {'MOKC': 142, 'MWC': 15, 'SH': 48}


class HospitalistSchedulerLP:
    """
    Build & solve a 28-day 3-hospital schedule with:
      - FT: 7 on / 7 off (unknown offset, enforced via one-hot cyclic template)
      - PT: no singletons (must work in blocks of >=2), optional monthly cap (e.g. PT4, PT7)
      - WD/WE: weekdays-only or weekends-only
      - Location eligibility per doctor
      - One location per day per doctor (no partial days)
    Objective: heavily penalize under-coverage, lightly penalize over-coverage,
               optionally penalize number of doctors used (lambda_used).
    """

    def __init__(self, **kwargs):
        # Core sets/params
        self.days = int(kwargs.get('days', 28))
        assert self.days == 28, "This model assumes a 28-day month."
        self.hospitals = kwargs.get('hospitals', list(DEFAULT_DEMAND.keys()))
        self.doctors = kwargs.get('doctors', DEFAULT_DOCTORS)

        # Demand and scaling (aka daily_demand multipliers)
        base_demand = kwargs.get('base_demand', DEFAULT_DEMAND)
        # Allow alias `daily_demand` for scaling factors
        demand_scale = kwargs.get(
            'demand_scale', kwargs.get('daily_demand', None))
        if demand_scale is None:
            demand_scale = {h: 1.0 for h in self.hospitals}
        self.req = {
            h: base_demand[h] * float(demand_scale.get(h, 1.0)) for h in self.hospitals}

        # Penalties
        self.pen_under = float(kwargs.get('pen_under', 100.0))
        self.pen_over = float(kwargs.get('pen_over',  1.0))
        # Optional tiny penalty to reduce unique doctors used (0 = ignore)
        self.lambda_used = float(kwargs.get('lambda_used', 0.01))

        # Calendar: day 0 is Monday by default; adjust via start_dow (0=Mon,...,6=Sun)
        self.start_dow = int(kwargs.get('start_dow', 0))

        # Internal containers (filled in build())
        self.model = None
        self.x = {}       # assignment vars: (doc,hosp,day) -> {0,1}
        self.y = {}       # working indicator: (doc,day) -> {0,1}
        self.u = {}       # under-coverage: (hosp,day) >= 0
        self.o = {}       # over-coverage:  (hosp,day) >= 0
        self.zft = {}     # FT pattern selection: (doc,offset) -> {0,1}
        self.used = {}    # doc used at least once: (doc) -> {0,1}

        # Precompute helper maps
        self.doc_names = list(self.doctors.keys())
        self.doc_index = {d: i for i, d in enumerate(self.doc_names)}
        self.allowed = self._build_allowed_map()
        self.patients = {d: int(self.doctors[d]['patients'])
                         for d in self.doc_names}
        self.is_ft, self.is_pt, self.only_wd, self.only_we, self.pt_cap = self._parse_types()

        self.weekday_set, self.weekend_set = self._weekday_weekend_sets()

        # Solution cache
        self.sol_status = None
        self._sol_cap = None  # capacity by (h,t)
        self._sol_under = None
        self._sol_over = None

    # ---------- Helpers ----------
    def _build_allowed_map(self):
        allowed = {d: [] for d in self.doc_names}
        for d, attrs in self.doctors.items():
            for h in self.hospitals:
                if attrs.get(h, 0) == 1:
                    allowed[d].append(h)
        return allowed

    def _parse_types(self):
        is_ft = {}
        is_pt = {}
        only_wd = {}
        only_we = {}
        pt_cap = {}
        for d, attrs in self.doctors.items():
            code = str(attrs.get('type', '')).upper()
            ft = 'FT' in code
            pt = 'PT' in code
            wd = 'WD' in code
            we = 'WE' in code
            # Extract first integer if present (PT7 => 7, PT4 => 4)
            num = ''.join([c for c in code if c.isdigit()])
            cap = int(num) if num else None

            is_ft[d] = ft
            # anyone not FT is treated PT-like for the "no singletons" rule if you prefer strict PT only
            is_pt[d] = pt or (not ft)
            only_wd[d] = wd
            only_we[d] = we
            pt_cap[d] = cap
        return is_ft, is_pt, only_wd, only_we, pt_cap

    def _weekday_weekend_sets(self):
        # day 0..27; weekday if dow in {0..4}, weekend if {5,6}
        weekdays, weekends = set(), set()
        for t in range(self.days):
            dow = (self.start_dow + t) % 7
            if dow in {5, 6}:
                weekends.add(t)
            else:
                weekdays.add(t)
        return weekdays, weekends

    def _ft_template(self, offset):
        """
        14-day cycle: 7 on, 7 off, repeated.
        offset in [0..13], day t is ON if ((t - offset) mod 14) in [0..6]
        Returns list length self.days of 0/1.
        """
        pat = []
        for t in range(self.days):
            on = 1 if ((t - offset) % 14) <= 6 else 0
            pat.append(on)
        return pat

    # ---------- Build ----------
    def build(self):
        T = range(self.days)
        Hs = list(self.hospitals)
        D = list(self.doc_names)

        self.model = pl.LpProblem("Hospitalist_Scheduling", pl.LpMinimize)

        # Variables
        # assignment x[d,h,t] only where allowed
        for d in D:
            for t in T:
                for h in Hs:
                    if h in self.allowed[d]:
                        self.x[(d, h, t)] = pl.LpVariable(
                            f"x_{self.doc_index[d]}_{h}_{t}", 0, 1, cat="Binary")
        # y[d,t] working indicator
        for d in D:
            for t in T:
                self.y[(d, t)] = pl.LpVariable(
                    f"y_{self.doc_index[d]}_{t}", 0, 1, cat="Binary")
        # under/over by hospital-day
        for h in Hs:
            for t in T:
                self.u[(h, t)] = pl.LpVariable(f"under_{h}_{t}", lowBound=0)
                self.o[(h, t)] = pl.LpVariable(f"over_{h}_{t}", lowBound=0)
        # doc used
        for d in D:
            self.used[d] = pl.LpVariable(
                f"used_{self.doc_index[d]}", 0, 1, cat="Binary")

        # FT pattern one-hot offsets
        for d in D:
            if self.is_ft[d]:
                for s in range(14):
                    self.zft[(d, s)] = pl.LpVariable(
                        f"zft_{self.doc_index[d]}_{s}", 0, 1, cat="Binary")

        # Constraints

        # 1) Link y to x and limit one location per day
        for d in D:
            for t in T:
                # sum_h x = y   (if doctor has no allowed hospitals this sums to 0)
                self.model += (
                    pl.lpSum(self.x[(d, h, t)] for h in Hs if (
                        d, h, t) in self.x) == self.y[(d, t)],
                    f"link_y_{d}_{t}"
                )
                # At most one location — implied by equality above, but keep for clarity
                self.model += (
                    pl.lpSum(self.x[(d, h, t)]
                             for h in Hs if (d, h, t) in self.x) <= 1,
                    f"one_loc_{d}_{t}"
                )

        # 2) WD/WE restrictions
        for d in D:
            if self.only_wd[d]:
                for t in self.weekend_set:
                    self.model += (self.y[(d, t)] == 0, f"wd_block_{d}_{t}")
            if self.only_we[d]:
                for t in self.weekday_set:
                    self.model += (self.y[(d, t)] == 0, f"we_block_{d}_{t}")

        # 3) FT pattern: select one offset, enforce 7-on/7-off template
        for d in D:
            if self.is_ft[d]:
                # choose exactly one offset
                self.model += (
                    pl.lpSum(self.zft[(d, s)] for s in range(14)) == 1,
                    f"ft_one_offset_{d}"
                )
                # y equals the chosen template
                for t in T:
                    tmpl = [self._ft_template(s)[t] for s in range(14)]
                    # y[d,t] == sum_s tmpl[s]*z_s   (exact equality)
                    self.model += (
                        self.y[(d, t)] == pl.lpSum(tmpl[s]*self.zft[(d, s)]
                                                   for s in range(14)),
                        f"ft_shape_{d}_{t}"
                    )

        # 4) PT “no singletons” & optional monthly cap
        for d in D:
            if self.is_pt[d] and not self.is_ft[d]:
                # Interior days: y_t <= y_(t-1) + y_(t+1)
                for t in range(1, self.days-1):
                    self.model += (
                        self.y[(d, t)] <= self.y[(d, t-1)] + self.y[(d, t+1)],
                        f"pt_no_singletons_mid_{d}_{t}"
                    )
                # Edges: no singletons at ends
                self.model += (self.y[(d, 0)] <= self.y[(d, 1)],
                               f"pt_no_singletons_start_{d}")
                self.model += (self.y[(d, self.days-1)] <=
                               self.y[(d, self.days-2)], f"pt_no_singletons_end_{d}")

                # Optional monthly cap: sum_t y <= cap
                if self.pt_cap[d] is not None:
                    self.model += (
                        pl.lpSum(self.y[(d, t)] for t in T) <= self.pt_cap[d],
                        f"pt_cap_{d}"
                    )

        # 5) Coverage balance per hospital-day: capacity - required = over - under
        for h in Hs:
            req_h = float(self.req[h])
            for t in T:
                cap = pl.lpSum(self.patients[d]*self.x[(d, h, t)]
                               for d in D if (d, h, t) in self.x)
                self.model += (
                    cap - req_h == self.o[(h, t)] - self.u[(h, t)],
                    f"balance_{h}_{t}"
                )

        # 6) Link y to doc-used
        for d in D:
            for t in T:
                self.model += (self.y[(d, t)] <=
                               self.used[d], f"used_link_{d}_{t}")

        # Objective
        obj = pl.lpSum(self.pen_under * self.u[(h, t)] + self.pen_over * self.o[(h, t)]
                       for h in Hs for t in T)
        if self.lambda_used > 0:
            obj += self.lambda_used * pl.lpSum(self.used[d] for d in D)

        self.model += obj

        return self

    # ---------- Solve ----------
    def solve(self, solver=None, msg=False, time_limit=None):
        """
        Solve the model. Returns pulp status string.
        - solver: optional pulp solver instance (e.g., pl.PULP_CBC_CMD(msg=1, timeLimit=60))
        - msg: True/False to show solver logs if using default CBC
        - time_limit: seconds (only if using default CBC)
        """
        if self.model is None:
            self.build()

        if solver is None:
            solver = pl.PULP_CBC_CMD(
                msg=msg, timeLimit=time_limit, gapRel=0.001)

        self.model.solve(solver)
        self.sol_status = pl.LpStatus[self.model.status]

        # Cache solution coverage numbers
        self._cache_solution_numbers()
        return self.sol_status

    def _cache_solution_numbers(self):
        if self.sol_status not in ("Optimal", "Feasible"):
            self._sol_cap, self._sol_under, self._sol_over = None, None, None
            return

        T = range(self.days)
        Hs = list(self.hospitals)

        # capacity per (h,t)
        cap = {(h, t): 0.0 for h in Hs for t in T}
        for (d, h, t), var in self.x.items():
            if var.varValue is not None and var.varValue > 0.5:
                cap[(h, t)] += self.patients[d]

        under = {(h, t): float(self.u[(h, t)].value() or 0.0)
                 for h in Hs for t in T}
        over = {(h, t): float(self.o[(h, t)].value() or 0.0)
                for h in Hs for t in T}

        self._sol_cap, self._sol_under, self._sol_over = cap, under, over

    # ---------- Metrics ----------
    def metrics(self, tol=1e-6):
        """
        Returns a dict of:
          - per_hospital: {h: {'pct_days_fully_covered', 'avg_pct_covered'}}
          - system: {'pct_days_all_sites_covered', 'avg_pct_covered'}
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."
        T = range(self.days)
        Hs = list(self.hospitals)

        per_h = {}
        sys_days_full = 0
        sys_avg_sum = 0.0

        for h in Hs:
            req = float(self.req[h])
            days_full = 0
            avg_sum = 0.0
            for t in T:
                cap = self._sol_cap[(h, t)]
                # fully covered?
                if cap + tol >= req:
                    days_full += 1
                # % covered for the day (cap/req, capped at 1.0)
                pct = 1.0 if req <= tol else min(1.0, cap / req)
                avg_sum += pct
            per_h[h] = {
                'pct_days_fully_covered': days_full / self.days,
                'avg_pct_covered': avg_sum / self.days
            }

        # System-level metrics (all sites simultaneously)
        for t in T:
            # all covered?
            all_ok = True
            num = 0.0
            den = 0.0
            for h in Hs:
                req = float(self.req[h])
                cap = self._sol_cap[(h, t)]
                if cap + tol < req:
                    all_ok = False
                num += cap
                den += req
            if all_ok:
                sys_days_full += 1
            sys_avg_sum += (1.0 if den <= tol else min(1.0, num/den))

        system = {
            'pct_days_all_sites_covered': sys_days_full / self.days,
            'avg_pct_covered': sys_avg_sum / self.days
        }

        return {'per_hospital': per_h, 'system': system}

    # ---------- Convenience extractors ----------
    def schedule_table(self):
        """
        Returns a nested dict: {t: {h: [doctors]}}
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."
        T = range(self.days)
        Hs = list(self.hospitals)
        sched = {t: {h: [] for h in Hs} for t in T}
        for (d, h, t), var in self.x.items():
            if (var.varValue or 0) > 0.5:
                sched[t][h].append(d)
        return sched

    def daily_coverage(self):
        """
        Returns coverage arrays per hospital: {h: [capacity_by_day]}, and required levels.
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."
        Hs = list(self.hospitals)
        by_h = {h: [self._sol_cap[(h, t)]
                    for t in range(self.days)] for h in Hs}
        return by_h, self.req.copy()

        # ---------- Visualization ----------
    def grid(self, show=True, figsize=None, return_data=False):
        """
        Visualize the solved schedule as a doctor x day grid (color = hospital assignment).
        - show: whether to call plt.show()
        - figsize: optional (w,h); defaults to (16, 0.45 * num_doctors)
        - return_data: if True, returns (schedule_grid, doctor_list, color_label_map)
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."

        T = list(range(self.days))
        Hs = list(self.hospitals)
        doctor_list = list(self.doc_names)
        day_count = len(T)
        doctor_count = len(doctor_list)

        # map hospitals to palette indices 1..|H|
        hospital_color_map = {h: i+1 for i, h in enumerate(Hs)}
        color_label_map = {0: 'Not Working'}
        color_label_map.update({i+1: h for i, h in enumerate(Hs)})

        schedule_grid = np.zeros((doctor_count, day_count), dtype=int)

        # fill grid using solved x[(d,h,t)]
        for i, d in enumerate(doctor_list):
            for t in T:
                for h in Hs:
                    key = (d, h, t)
                    if key in self.x:
                        v = self.x[key].varValue or 0.0
                        if v > 0.5:
                            schedule_grid[i, t] = hospital_color_map[h]
                            break  # one location per day

        if figsize is None:
            figsize = (16, max(4, doctor_count * 0.45))

        cmap = plt.get_cmap('viridis', len(Hs) + 1)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(schedule_grid, aspect='auto',
                       cmap=cmap, vmin=0, vmax=len(Hs))

        ax.set_xticks(np.arange(day_count))
        ax.set_xticklabels(
            [f'Day {d+1}' for d in range(day_count)], rotation=90)
        ax.set_yticks(np.arange(doctor_count))
        ax.set_yticklabels(doctor_list)

        ax.set_xticks(np.arange(-.5, day_count, 1), minor=True)
        ax.set_yticks(np.arange(-.5, doctor_count, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.25)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap(i), label=label)
                           for i, label in color_label_map.items()]
        ax.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(1.15, 1.0))

        ax.set_title("Doctor Schedule Grid (Color = Hospital Assignment)")
        plt.tight_layout()
        if show:
            plt.show()

        if return_data:
            return schedule_grid, doctor_list, color_label_map

    # ---------- Auditing ----------
    def audit(self, print_ok=True, tol=1e-6):
        """
        Prints only doctors with violations (or 'no violations found').
        Checks:
          - One location per day / link to y
          - Allowed hospital assignment
          - WD/WE restrictions
          - PT monthly caps (e.g., PT4/7) and 'no singletons'
          - FT 7-on/7-off template (total 14 days on; pattern matches selected offset)
        Returns a dict {doctor: [violations...]}.
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."

        T = list(range(self.days))
        Hs = list(self.hospitals)
        violations_by_doc = {}
        weekdays = self.weekday_set
        weekends = self.weekend_set

        # helper: chosen FT offset (if any)
        def chosen_offset(d):
            if not self.is_ft[d]:
                return None
            for s in range(14):
                z = self.zft[(d, s)].varValue or 0.0
                if z > 0.5:
                    return s
            return None  # unsolved/degenerate

        for d in self.doc_names:
            viols = []

            # collect daily x, y
            yvec = [float(self.y[(d, t)].varValue or 0.0) for t in T]
            x_by_day = []
            for t in T:
                day_assign = {h: 0.0 for h in Hs}
                for h in Hs:
                    key = (d, h, t)
                    if key in self.x:
                        day_assign[h] = float(self.x[key].varValue or 0.0)
                x_by_day.append(day_assign)

            # --- one location per day and link to y ---
            for t in T:
                s = sum(x_by_day[t].values())
                if s > 1 + tol:
                    viols.append(
                        f"multiple locations on day {t+1} (sum={s:.2f})")
                # equality link: sum_h x == y
                if abs(s - yvec[t]) > 1e-3:  # tolerate tiny LP fuzz
                    viols.append(
                        f"link mismatch on day {t+1} (sum_x={s:.2f} != y={yvec[t]:.2f})")

            # --- allowed hospitals ---
            allowed_h = set(self.allowed[d])
            for t in T:
                for h, v in x_by_day[t].items():
                    if v > 0.5 and h not in allowed_h:
                        viols.append(
                            f"worked at disallowed hospital {h} on day {t+1}")

            worked_days = [t for t in T if yvec[t] > 0.5]
            total_worked = len(worked_days)

            # --- WD / WE restrictions ---
            if self.only_wd[d]:
                bad = [t for t in worked_days if t in weekends]
                if bad:
                    viols.append(
                        f"worked on weekend (days {[b+1 for b in bad]}) but WD-only")
            if self.only_we[d]:
                bad = [t for t in worked_days if t in weekdays]
                if bad:
                    viols.append(
                        f"worked on weekday (days {[b+1 for b in bad]}) but WE-only")

            # --- PT caps & no-singletons ---
            if self.is_pt[d] and not self.is_ft[d]:
                cap = self.pt_cap[d]
                if cap is not None and total_worked > cap + tol:
                    viols.append(
                        f"exceeded monthly cap {cap} (worked {total_worked})")
                # no singletons: if y[t]=1 then y[t-1]=1 or y[t+1]=1
                # handle edges
                if self.days >= 2:
                    if yvec[0] > 0.5 and yvec[1] < 0.5:
                        viols.append("singleton at day 1")
                    if yvec[-1] > 0.5 and yvec[-2] < 0.5:
                        viols.append(f"singleton at day {self.days}")
                    for t in range(1, self.days - 1):
                        if yvec[t] > 0.5 and (yvec[t-1] < 0.5 and yvec[t+1] < 0.5):
                            viols.append(f"singleton at day {t+1}")

            # --- FT 7/7 template checks ---
            if self.is_ft[d]:
                # (a) total 14 on-days over 28
                if total_worked != 14:
                    viols.append(
                        f"FT should work 14 days; works {total_worked}")
                # (b) pattern matches chosen offset
                s = chosen_offset(d)
                if s is not None:
                    tmpl = self._ft_template(s)
                    for t in T:
                        if int(yvec[t] > 0.5) != tmpl[t]:
                            viols.append(
                                f"FT template mismatch on day {t+1} (offset={s})")
                            break
                else:
                    viols.append("FT has no chosen offset (zft not set)")

            if viols:
                violations_by_doc[d] = viols

        if violations_by_doc:
            for d, vs in violations_by_doc.items():
                print(d + ":")
                for v in vs:
                    print("  - " + v)
        elif print_ok:
            print("no violations found")

        return violations_by_doc

        # ---------- Coverage visualization ----------
    def plot_coverage(self, show=True, figsize=(14, 9), day_labels=True):
        """
        Plot required vs scheduled capacity for each hospital
        as stacked subplots in one figure.
        """
        assert self._sol_cap is not None, "No solution cached. Call solve() first."

        T = list(range(self.days))
        x = np.arange(self.days)

        fig, axes = plt.subplots(
            len(self.hospitals), 1, figsize=figsize, sharex=True)

        if len(self.hospitals) == 1:
            axes = [axes]  # make iterable if only 1 hospital

        for ax, h in zip(axes, self.hospitals):
            req = float(self.req[h])
            cap = np.array([self._sol_cap[(h, t)] for t in T], dtype=float)
            under = np.array([self._sol_under[(h, t)] for t in T], dtype=float)
            over = np.array([self._sol_over[(h, t)] for t in T], dtype=float)

            # under / over bars
            ax.bar(x, under, alpha=0.6, label='UNDER',
                   edgecolor='none', color='tab:red')
            ax.bar(x, over, alpha=0.6, bottom=0, label='OVER',
                   edgecolor='none', color='tab:green')

            # scheduled capacity line
            ax.plot(x, cap, marker='o', linewidth=1.5,
                    label='Scheduled', color='tab:blue')

            # required horizontal line
            ax.axhline(req, linestyle='--', linewidth=1.25,
                       label='Required', color='black')

            # cosmetics
            ax.set_ylabel("Patients")
            ax.set_title(f"{h}")
            ax.grid(True, axis='y', linewidth=0.3)
            ax.legend(loc='upper right')

        # shared x-axis labels
        if day_labels:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels([f"Day {t+1}" for t in T], rotation=90)
        else:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels([t+1 for t in T])

        axes[-1].set_xlabel("Day")
        fig.suptitle("Coverage — All Hospitals", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if show:
            plt.show()

    # Add these two methods inside HospitalistSchedulerLP

    def solve_mip(self, msg=False, time_limit=None, rel_gap=None, solver=None):
        """Solve the integer model (your production schedule)."""
        if self.model is None:
            self.build()
        if solver is None:
            try:
                solver = pl.PULP_CBC_CMD(msg=msg, timeLimit=time_limit,
                                         gapRel=(rel_gap if rel_gap is not None else None))
            except TypeError:
                opts = [] if rel_gap is None else [f"ratioGap={rel_gap}"]
                solver = pl.PULP_CBC_CMD(
                    msg=msg, timeLimit=time_limit, options=opts)
        self.model.solve(solver)
        self.sol_status = pl.LpStatus[self.model.status]
        self._cache_solution_numbers()
        return self.sol_status

    def solve_lp_duals(self, msg=False, time_limit=None, solver=None):
        """
        Solve the LP relaxation (temporarily relax binaries to continuous) and
        return (shadow_prices, reduced_costs). Also caches them on self._lp_duals/_lp_rc.
        Does NOT change your final integer schedule (we restore integrality afterward).
        """
        if self.model is None:
            self.build()

        # choose solver (any LP solver is fine; CBC works)
        if solver is None:
            solver = pl.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)

        # relax integrality in-place, remember original categories
        orig_cat = {v.name: v.cat for v in self.model.variables()}
        for v in self.model.variables():
            v.cat = pl.LpContinuous

        # solve LP
        self.model.solve(solver)
        lp_status = pl.LpStatus[self.model.status]

        # grab duals (shadow prices) and reduced costs
        self._lp_duals = {name: c.pi for name,
                          c in self.model.constraints.items()}
        self._lp_slacks = {name: c.slack for name,
                           c in self.model.constraints.items()}
        self._lp_rc = {v.name: v.dj for v in self.model.variables()}

        # restore integrality
        for v in self.model.variables():
            v.cat = orig_cat.get(v.name, v.cat)

        # (optional) do not re-solve the MIP here; you’ll call solve_mip() when you want the schedule

        # package handy outputs
        shadow_prices = {(h, t): self._lp_duals[f"balance_{h}_{t}"]
                         for h in self.hospitals for t in range(self.days)}
        reduced_costs = {(d, h, t): self._lp_rc[self.x[(d, h, t)].name]
                         for (d, h, t) in self.x}

        return lp_status, shadow_prices, reduced_costs
