#!/usr/bin/env python
"""Standalone OpenVSP mesh generation script.

Runs under the openvsp310 conda env (Python 3.10) because the openvsp
_vsp.so binary is linked against libpython3.10.

Usage:
    python _vsp_generate_stl.py <model.vsp3> <output.stl> [B1=val B2=val ...]

Outputs an STL file with coordinates in meters (vsp3 is in mm, divided by 1000).
"""
import sys
import os
import json
import tempfile

import openvsp as vsp


def generate(vsp3_path, stl_out, params):
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(vsp3_path)

    user_parms = vsp.GetAllUserParms()
    planform = {}
    for pid in user_parms:
        name = vsp.GetParmName(pid)
        group = vsp.GetParmGroupName(pid)
        if group == "Planform":
            planform[name] = pid

    for name, val in params.items():
        if name in planform:
            vsp.SetParmVal(planform[name], float(val))
    vsp.Update()

    vsp.ExportFile(stl_out, vsp.SET_ALL, vsp.EXPORT_STL)

    if not os.path.exists(stl_out) or os.path.getsize(stl_out) == 0:
        print(json.dumps({"error": "STL export failed"}))
        sys.exit(1)

    print(json.dumps({"stl_path": stl_out, "size": os.path.getsize(stl_out)}))


if __name__ == "__main__":
    vsp3_path = sys.argv[1]
    stl_out = sys.argv[2]
    params = {}
    for arg in sys.argv[3:]:
        k, v = arg.split("=", 1)
        params[k] = float(v)
    generate(vsp3_path, stl_out, params)
