import fanta_rust as fanta
import math
import time

def render(width, height):
    w = float(width)
    h = float(height)

    # Main Layout
    fanta.Column().size(w, h).bg(fanta.Color(0.02, 0.02, 0.03, 1.0)).padding(20).gap(20)
    
    with fanta.Row().gap(15).align(fanta.Align.Center):
        fanta.Text("Advanced Color Grading").font_size(24).color(fanta.Color(1.0, 0.8, 0.2, 1))
        fanta.Text("Curve Editor Beta").font_size(14).color(fanta.Color(0.5, 0.5, 0.6, 1))

    with fanta.Row().gap(20).width_fill().height_fill():
        with fanta.Column().flex_grow(1.0).gap(15):
            fanta.Text("Master RGB").font_size(16)
            fanta.CurveEditor("master_curve").height(300).build()
            
            with fanta.Row().gap(10).height(200):
                with fanta.Column().flex_grow(1):
                    fanta.Text("Red Channel").font_size(14)
                    fanta.CurveEditor("red_curve").height(150).build()
                with fanta.Column().flex_grow(1):
                    fanta.Text("Green Channel").font_size(14)
                    fanta.CurveEditor("green_curve").height(150).build()
                with fanta.Column().flex_grow(1):
                    fanta.Text("Blue Channel").font_size(14)
                    fanta.CurveEditor("blue_curve").height(150).build()
        
        # --- Right Panel: 3D Preview & LUTs ---
        with fanta.Column().width(400).bg(fanta.Color(0.05, 0.05, 0.07, 1.0)).padding(15).gap(15):
            fanta.Text("3D Preview").font_size(18)
            
            # 3D Viewport with simulated camera and optional LUT
            t = time.time()
            cam_x = math.sin(t * 0.5) * 6.0
            cam_z = math.cos(t * 0.5) * 6.0
            
            # Using placeholder LUT ID 0 (no LUT) for now, but wiring controls
            lut_strength = fanta.get_state("lut_strength", 1.0)
            
            fanta.Scene3D("preview_3d") \
                .height(300) \
                .camera((cam_x, 2.0, cam_z), (0.0, 0.0, 0.0)) \
                .lut(0, lut_strength) \
                .build()
                
            fanta.Text("LUT Controls").font_size(16)
            fanta.Slider("lut_strength", 0.0, 1.0).height(30).build()
            
            with fanta.Row().gap(10):
                if fanta.Button("Load .CUBE").height(40).build():
                    print("Load CUBE clicked")
                if fanta.Button("Export .CUBE").height(40).build():
                    print("Export CUBE clicked - TODO: Generate from Curves")
            
            fanta.Text("Presets").font_size(16)
            with fanta.Column().gap(8):
                if fanta.Button("Linear").height(30).build():
                    print("Apply Linear")
                if fanta.Button("S-Curve").height(30).build():
                    print("Apply S-Curve")
                if fanta.Button("Teal & Orange").height(30).build():
                    print("Apply Teal & Orange")

    fanta.End()
