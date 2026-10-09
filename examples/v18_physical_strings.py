import genny

props = genny.StringPhysicalProperties(
    length_m=.65,
    radius_m=.00035,
    material='steel',
    frequency_hz=440.,
)
print('tension N:', props.tension_n)
print('Z N*s/m:', props.characteristic_impedance_n_s_m)
print('B:', props.inharmonicity_B)
print('longitudinal Hz:', props.longitudinal_frequency_hz)

string = genny.DispersiveString(props, sr=48000)
audio = string.render_pluck(.5, velocity=.8, position=.22)
print('samples:', len(audio), 'peak:', max(abs(audio)))
