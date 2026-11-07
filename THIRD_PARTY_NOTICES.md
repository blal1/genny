# Third-party notices

## Klang / procedural physical-model engines

Parts of `genny/physical/` are adapted from the user-supplied **procedural** project by Chris Nash,
Copyright © 2025 Chris Nash, under the **Klang Open License 1.0** (based on Apache 2.0).
The complete license is included as `THIRD_PARTY_KLANG_LICENSE.txt` and
`genny/physical/KLANG_LICENSE.txt`.

Since genny 1.0, `genny/klang.py` (vehicle, interface, body and world sounds translated to Python),
and functions marked as ports in `genny/creatures.py`, `genny/choir.py`, `genny/compose.py` and
`genny/foley.py`, are also derived from that project and fall under the same licence.

The adapted files were reorganized under `genny.physical`, package imports were changed for Genny,
and a stable high-level API was added in `genny.physics`, `genny.acoustics`, and `genny.identify`.

Important: the Klang Open License adds an attribution requirement for interactive audio-visual
products. Consult the bundled license before shipping a game, simulation, or multimedia installation.

## genny/cyber.py: models and constants read in other projects

`genny/cyber.py` contains no code copied from another project. It re-implements published models, and
some of its numeric constants were read in these sources:

- `pipe_engine`: the engine model of Baldan, Lachambre, Delle Monache & Boussard (2015) as implemented in
  **DasEtwas/enginesound** and **Antonio-R1/engine-sound-generator** (both MIT); reflection coefficients,
  mix levels and muffler lengths are those of DasEtwas' `default.esc`.
- `cyber_ui`: the layer grammar and the note, gap and envelope ranges of **m1ckc3s/procedural-sounds** (MIT).
- `bearing` and `identify_machine`: constants and thresholds were set by measuring three sets of recordings,
  none of which is redistributed: the FSTF sound dataset (Mendeley Data n9y9c7xrz3), the AHU parabolic
  acoustic mirror bearing dataset (Lab-of-AMFD, no licence stated) and SUBF v2.0 (Kaggle, CC BY-NC-SA 4.0).
- `paulstretch`, `beat_repeat`, `waveset`: the algorithms of Paul Nasca's paulstretch and of
  **Soundpipe** (MIT) and **arraypress/paulstretch** (MIT).
- `pass_by`: the Harmonoise car coefficients (Jonasson et al., "Source modelling of road vehicles", 2004),
  transcribed from the table in **boschresearch/acoustic-traffic-simulation-counting** (GPL-3; the numbers
  are those of the public report).
- `turbo`, `tyre_squeal`, the wind term of `pass_by` and the default `ground` coefficient of `doppler`:
  formulas and constants read in **Agent00PED/ice-simulator**, which carries **no licence file**. Ask its
  author, or replace these constants, before a release that must be licence-clean.
- `texturize`: the model of McDermott & Simoncelli (Neuron, 2011), reduced.

## genny/chiptune.py: the Yamaha YM2612 core is a port of ymfm

The FM section of `genny/chiptune.py` (`_opn_core`, `_opn_vol` and the `OPN_*` tables) is a Python port of parts
of **ymfm** by Aaron Giles (`ymfm_fm.ipp`, `ymfm_opn.cpp`), used under the BSD 3-Clause License:

    Copyright (c) 2021, Aaron Giles. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided
    that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and
       the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
       the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
       promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

The NES and Game Boy sections contain no code from another project: they implement the hardware as described
in the NESdev wiki and in Pan Docs. The FM patches (`FM_PATCHES`) and the wave shapes (`GB_WAVES`) are genny's own.

## genny/chiptune2.py: sources read

- `ym2151`, `ym2610`: the FM engine is the ymfm port above. `OPM_STEP` is ymfm's table of YM2151 phase steps, which
  ymfm credits to David Viens' analysis of the chip (used there with his permission); the second-detune offsets and
  the key-code arithmetic are ymfm's (`ymfm_fm.ipp`, `ymfm_opm.cpp`, BSD 3-Clause, notice above).
- `sid`: no code is taken from another project. The rate-counter periods, the levels at which the decay slows, the
  noise register's taps and output bits and the 8580 resonance table were read in **reSID** by Dag Lem (GPL);
  they are measurements of the chip.
- `pokey`: no code is taken. The polynomial counter formulas, the dividers and the AUDC logic were read in MAME's
  `pokey.cpp` (Brad Oliver, Eric Smith, Juergen Buchmueller; BSD 3-Clause), which credits Perry McFarlane and the
  Altirra Hardware Reference Manual.
- `sn76489`: written from the SMS Power development wiki page "SN76489". `dpcm`: from the NESdev wiki, "APU DMC".
- `stk_fm`: a port of the four-operator FM voices of **The Synthesis ToolKit in C++ (STK)** (`FM.cpp`, `HevyMetl`,
  `PercFlut`, `BeeThree`, `Rhodey`, `Wurley`, `TubeBell`): their gain and sustain tables, operator ratios, envelopes
  and per-sample algorithms. The `fwavblnk.raw` wave is computed by its formula (same 256 values), not copied.

      The Synthesis ToolKit in C++ (STK)
      Copyright (c) 1995-2023 Perry R. Cook and Gary P. Scavone

      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
      documentation files (the "Software"), to deal in the Software without restriction, including without limitation
      the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
      to permit persons to whom the Software is furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all copies or substantial portions
      of the Software.

      Any person wishing to distribute modifications to the Software is asked to send the modifications to the
      original developer so that they can be incorporated into the canonical version. This is, however, not a
      binding provision of this license.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
      THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
      CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
      DEALINGS IN THE SOFTWARE.

## genny/retro.py: sfxr (jsfxr), Bfxr2, ZzFX, ZzFXM instruments and AKWF waves

- `sfxr_core`, the sfxr presets and the knob conversions are a port of **jsfxr** (`sfxr.js`), Eric Fredricksen's
  HTML5 port of DrPetter's sfxr. jsfxr is released into the public domain under **The Unlicense**
  (`jsfxr/UNLICENSE`), so no notice is required.

- The Bfxr extras (waves 4-11, overtones, the two pitch jumps, bit crush), the Bfxr preset generators and
  `jump_knob` are ported from **Bfxr2**, MIT License:

      Copyright (c) 2021 Stephen Lavelle

      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
      documentation files (the "Software"), to deal in the Software without restriction, including without limitation
      the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
      to permit persons to whom the Software is furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all copies or substantial portions
      of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
      THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
      CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
      DEALINGS IN THE SOFTWARE.

  Bfxr2's synthesis file `js/audio/Bfxr_DSP.js`, from which the extras are read, carries its own notice: it "uses
  ported/modified code from Thomas Vian's SfxrSynth", **Copyright 2010 Thomas Vian, Apache License, Version 2.0**
  (full text at the end of this file). genny's Python port of those parts is a modified version: the code was
  translated from JavaScript to Python/Numba and restructured.

- `zzfx_core` (`buildSamples`) and the ZzFX slot defaults are a port of **ZzFX** 1.3.2 (`ZzFX.js`), MIT License:

      Copyright (c) 2019 - Frank Force

      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
      documentation files (the "Software"), to deal in the Software without restriction, including without limitation
      the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
      to permit persons to whom the Software is furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all copies or substantial portions
      of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
      THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
      CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
      DEALINGS IN THE SOFTWARE.

- `ZZFXM_INSTRUMENTS` is the instrument table of **ZzFXM** (`tools/convert/lib/instruments.js`), MIT License,
  Copyright (c) 2020 Keith Clark:

      Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
      documentation files (the "Software"), to deal in the Software without restriction, including without limitation
      the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
      to permit persons to whom the Software is furnished to do so, subject to the following conditions:

      The above copyright notice and this permission notice shall be included in all copies or substantial portions
      of the Software.

      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
      THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
      CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
      DEALINGS IN THE SOFTWARE.

- `AKWF` (Bfxr's Rasp, FMSyn and Voice waves) holds `AKWF_granular_0044`, `AKWF_fmsynth_0012` and
  `AKWF_hvoice_0012` from the Adventure Kid Waveforms, as converted by Brad Roy and shipped in Bfxr2's `AKWF.js`;
  they are public domain (CC0 1.0), so no notice is required.

## genny/vehicle.py: PTR exhaust resonators

`PTR_EXHAUST` holds nine numbers per exhaust (a delay and two feedback coefficients for each of three resonators)
read from the trained checkpoints of **ptr-model** by Robin Doerfler and Lonce Wyse
(github.com/rdoerfler/ptr-model, "Physics-Informed Neural Engine Sound Modeling with Differentiable Pulse-Train
Synthesis", arXiv:2603.09391), released under CC BY-NC 4.0. No code and no network weights are included; the pulse
and resonator structure is written from the paper.

## genny/physical/nonverbal.py: soundgen presets

`PRESETS` (laugh, scream, cry, moan) are the parameter values of the M1 and F1 presets in `presets.R` of
**soundgen** by Andrey Anikin (GPL >= 2), and the bandwidth formula is its `getBandwidth`. No code is taken; the
synthesis is written from Anikin, "Soundgen: an open-source tool for synthesizing nonverbal vocalizations",
Behav. Res. Methods 51, 778-792 (2019) and the package's parameter documentation.

## Apache License, Version 2.0 (Thomas Vian's SfxrSynth, in Bfxr2's Bfxr_DSP.js)

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```
