"""Lumped habitat CO2/thermal mass balance. Engineering scenarios, not life-support control."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class Habitat:
 volume_m3:float=40.;ventilation_m3_h:float=120.;outside_co2_ppm:float=420.;co2_production_m3_person_h:float=.018;thermal_capacity_J_K:float=200000.;wall_UA_W_K:float=100.;air_rho_cp_J_m3_K:float=1200.
 def __post_init__(self):
  if not all(math.isfinite(v) and v>0 for v in self.__dict__.values()):raise ValueError('habitat physical configuration')
 def advance(self,co2_ppm,temp_C,occupants,outside_C,equipment_W,cooling_W,hours):
  if not all(math.isfinite(v) for v in [co2_ppm,temp_C,occupants,outside_C,equipment_W,cooling_W,hours]) or min(co2_ppm,occupants,equipment_W,cooling_W,hours)<0:raise ValueError('habitat input')
  # dC/dt = Q/V(Cout-C) + N*G/V*1e6, with t in hours.
  rate=self.ventilation_m3_h/self.volume_m3;equilibrium=self.outside_co2_ppm+occupants*self.co2_production_m3_person_h/self.ventilation_m3_h*1e6;co2=equilibrium+(co2_ppm-equilibrium)*math.exp(-rate*hours)
  conductance=self.wall_UA_W_K+self.air_rho_cp_J_m3_K*self.ventilation_m3_h/3600;net_heat=occupants*100+equipment_W-cooling_W;target=outside_C+net_heat/conductance;temperature=target+(temp_C-target)*math.exp(-conductance*hours*3600/self.thermal_capacity_J_K)
  return dict(co2_ppm=co2,temperature_C=temperature,origin='engineering_simulation',scope='mass/energy-balance forecast with stated assumptions; not observed sensor values',parameters=self.__dict__)
