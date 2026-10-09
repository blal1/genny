"""Genny 0.19: plain, wound and bowed strings using one common core."""
import genny

plain = genny.StringPhysicalProperties.from_frequency(196.0, material="steel")
wound = genny.WoundStringPhysicalProperties.from_frequency(
    55.0, length_m=0.864, core_material="steel", winding_material="nickel"
)

guitar = genny.DispersiveString(plain, sr=48000).render_pluck(1.5, velocity=0.8)
bass = genny.DispersiveString(wound, sr=48000).render_pluck(2.0, velocity=0.8)
bowed = genny.DispersiveString(plain, sr=48000).render_bow(1.5, bow_velocity=0.18, bow_pressure=0.75)

print("plain tension N:", plain.tension_n)
print("wound linear density kg/m:", wound.linear_density_kg_m)
print("wound helix factor:", wound.helix_length_factor)
print("wound B:", wound.inharmonicity_B)
