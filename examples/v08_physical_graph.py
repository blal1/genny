"""Small Genny 0.8 physical-model examples."""
import numpy as np
import genny

# Explicit mechanical port and passive coupling.
n = 256
a = genny.PhysicalPort.mechanical(np.ones(n), np.full(n, .1), "driver")
b = genny.PhysicalPort.mechanical(np.zeros(n), np.zeros(n), "body")
g = genny.PhysicalGraph(); g.add("driver", a); g.add("body", b); g.couple("driver", "body", .7)
ports = g.process()
print("coupled power samples:", ports["body"].power[:4])

print("motor commutation Hz:", genny.ElectricMotor(rpm=6000).commutation_hz)
print("gear mesh Hz:", genny.GearTrain(rpm=1800, driver_teeth=20).mesh_hz)
print("kick samples:", len(genny.KickDrum().strike(.3)))
print("snare samples:", len(genny.SnareDrum().strike(.3)))
print("explosion samples:", len(genny.Explosion(distance_m=5).render(.8)))
