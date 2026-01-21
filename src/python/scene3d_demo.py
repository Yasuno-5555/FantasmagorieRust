import fanta_rust as fanta
import math
import time

def render(width, height):
    # Persistent state for camera
    if not hasattr(render, "cam_rot"):
        render.cam_rot = [0.4, 0.4] # pitch, yaw
        render.cam_dist = 5.0
        render.last_time = time.time()

    w = float(width)
    h = float(height)
    t = time.time()
    dt = t - render.last_time
    render.last_time = t

    # Main Layout
    fanta.Column().size(w, h).bg(fanta.Color(0.02, 0.02, 0.03, 1.0)).padding(20).gap(15)
    
    with fanta.Row().gap(15).align(fanta.Align.Center):
        fanta.Text("3D Integration Layer").font_size(28).color(fanta.Color(0.4, 0.8, 1.0, 1))
        fanta.Text("Phase 3").font_size(14).color(fanta.Color(0.5, 0.5, 0.6, 1))

    with fanta.Row().gap(20).width_fill().height_fill():
        # --- 3D Viewport ---
        # In a real app, you would render your 3D scene to a texture 
        # using an external engine (wgpu, taichi, etc.) and pass the texture ID.
        # Here we use the placeholder (texture_id=0) or we could simulate one.
        viewport = fanta.Scene3D("main_viewport").height(500)
        
        # Interactive Orbit Controls
        if viewport.active():
            dx, dy = fanta.mouse_delta()
            # Orbit logic
            render.cam_rot[1] -= dx * 0.005 # Yaw
            render.cam_rot[0] = max(-1.5, min(1.5, render.cam_rot[0] + dy * 0.005)) # Pitch
        
        # Auto-rotate if not active
        if not viewport.active() and not viewport.hovered():
            render.cam_rot[1] += dt * 0.2

        p, y = render.cam_rot
        d = render.cam_dist
        cx = d * math.cos(p) * math.sin(y)
        cy = d * math.sin(p)
        cz = d * math.cos(p) * math.cos(y)
        
        viewport.camera((cx, cy, cz), (0.0, 0.0, 0.0)).fov(45.0).build()

        # --- Sidebar (Settings) ---
        with fanta.Column().width(300).gap(20).padding(15).bg(fanta.Color(0.1, 0.1, 0.12, 0.8)).radius(10):
            fanta.Text("Viewport Settings").font_size(20)
            
            fanta.Text(f"Camera Position").font_size(14).color(fanta.Color(0.6, 0.6, 0.6, 1))
            fanta.Text(f"X: {cx:.2f}\nY: {cy:.2f}\nZ: {cz:.2f}").font_size(16)
            
            if fanta.Button("Reset View").clicked():
                render.cam_rot = [0.4, 0.4]
                
            fanta.Box().height(20)
            fanta.Text("Instructions:").font_size(14).color(fanta.Color(0.7, 0.5, 0.3, 1))
            fanta.Text("• Drag viewport to orbit\n• Integrates with external FBOs").font_size(12)

    fanta.End() # End Root
