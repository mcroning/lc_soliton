from __future__ import annotations

import json
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display


class PrdataEditor:
    def __init__(self, preset_path):
        self.preset_path = Path(preset_path)
        self.prdata = json.loads(self.preset_path.read_text())

        self.P = widgets.FloatText(value=float(self.prdata.get("P", 1.0)), description="P mW")
        self.bias = widgets.FloatText(value=float(self.prdata.get("bias_voltage", 1.0)), description="bias V")
        self.rlen = widgets.FloatText(value=float(self.prdata.get("rlen", 500.0)), description="rlen um")
        self.dz = widgets.FloatText(value=float(self.prdata.get("dz", 5.0)), description="dz um")
        self.tend = widgets.FloatText(value=float(self.prdata.get("tend", 0.01)), description="tend")
        self.tsteps = widgets.IntText(value=int(self.prdata.get("tsteps", 5)), description="tsteps")
        self.mode = widgets.Dropdown(
            options=["Static", "Time Dependent"],
            value=str(self.prdata.get("time_behavior", "Time Dependent")),
            description="mode",
        )

        self.out = widgets.Output()

    def update_prdata(self):
        self.prdata["P"] = float(self.P.value)
        self.prdata["bias_voltage"] = float(self.bias.value)
        self.prdata["rlen"] = float(self.rlen.value)
        self.prdata["dz"] = float(self.dz.value)
        self.prdata["tend"] = float(self.tend.value)
        self.prdata["tsteps"] = int(self.tsteps.value)
        self.prdata["time_behavior"] = str(self.mode.value)
        return self.prdata

    def show(self):
        btn = widgets.Button(description="Update prdata", button_style="success")

        def _click(_):
            with self.out:
                self.out.clear_output()
                self.update_prdata()
                print(json.dumps(self.prdata, indent=2))

        btn.on_click(_click)

        display(
            widgets.VBox([
                widgets.HBox([self.P, self.bias]),
                widgets.HBox([self.rlen, self.dz]),
                widgets.HBox([self.tend, self.tsteps]),
                self.mode,
                btn,
                self.out,
            ])
        )

        return self