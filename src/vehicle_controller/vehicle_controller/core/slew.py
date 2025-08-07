import numpy as np
import copy

class SlewRate():
    def __init__(self, dt, v_max, a_max):
        self.dt = dt
        self.v_max = v_max
        self.a_max = a_max

        self.sp_last = None
        self.sp_last2 = None

    def slew_rate(self, current_pos, sp_target):
        if self.sp_last is None:
            self.sp_last = copy.copy(current_pos)
            self.sp_last2 = copy.copy(current_pos)

        sp_curr = copy.copy(sp_target)
        
        v_sp = (sp_curr - self.sp_last)/self.dt
        v_norm = np.linalg.norm(v_sp)

        if v_norm > self.v_max:
            v_sp = (v_sp / v_norm) * self.v_max
            sp_curr = self.sp_last + v_sp*self.dt

        a_sp = (sp_curr - 2* self.sp_last + self.sp_last2) / (self.dt**2)
        a_norm = np.linalg.norm(a_sp)

        if a_norm > self.a_max:
            a_sp = (a_sp / a_norm) * self.a_max
            sp_curr = 2*self.sp_last - self.sp_last2 + a_sp * self.dt**2


        dist_to_target = np.linalg.norm(current_pos - sp_target)
        # Supposed to be from 2as = v^2 but reduce it by a factor of 2 sqrt 2 just to be on the safe side.
        v_stop_max = 0.5 * np.sqrt(self.a_max * dist_to_target)

        v_sp = (sp_curr - self.sp_last) / self.dt
        v_norm = np.linalg.norm(v_sp)

        if v_norm > v_stop_max:
            v_sp = (v_sp / v_norm) * v_stop_max
            sp_curr = self.sp_last + v_sp*self.dt

        self.sp_last2 = copy.copy(self.sp_last)
        self.sp_last = copy.copy(sp_curr)

        return sp_curr
    
    def reset(self):
        self.sp_last = None
        self.sp_last2 = None