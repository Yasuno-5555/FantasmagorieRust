
import fantasmagorie as fanta
import math

class DemoApp:
    def __init__(self):
        self.snap_enabled = True
        self.current_frame = 0
        
    def build(self):
        # Update animation state
        self.current_frame += 1
        t = self.current_frame * 0.01

        # Root: Dark Theme
        fanta.Window(1280, 800)
        
        # Main Layout: Row
        fanta.Row()
        
        # --- Sidebar (Tools) ---
        fanta.Column()
        fanta.Width(300.0)
        fanta.BgColor(0.1, 0.1, 0.12, 1.0)
        fanta.Padding(10.0, 10.0, 10.0, 10.0)
        
        fanta.Text("Pro Tools Demo", 24.0)
        fanta.Height(40.0)
        
        # Curve Editor
        fanta.Text("Color Grading", 16.0)
        fanta.Height(24.0)
        fanta.CurveEditor()
        fanta.Height(200.0) # Explicit height for CurveEditor
        fanta.BgColor(0.15, 0.15, 0.18, 1.0)
        
        fanta.Height(20.0)
        
        # Snapping Control
        fanta.Text(f"Snapping: {'ON' if self.snap_enabled else 'OFF'}", 16.0)
        if fanta.Button("Toggle Snap"):
            self.snap_enabled = not self.snap_enabled
        
        fanta.Height(20.0)
        
        # Math Showcase
        fanta.Text("Physics Engine", 16.0)
        fanta.Height(24.0)
        fanta.Math(r"F = G \frac{m_1 m_2}{r^2}")
        fanta.Height(50.0)
        fanta.Math(r"\Delta v = \sqrt{2gh}")
        
        fanta.End() # Sidebar Column
        
        # --- Main Viewport ---
        fanta.Column()
        fanta.FlexGrow(1.0)
        
        # Top Ruler
        fanta.Ruler()
        fanta.Height(30.0)
        fanta.Orientation(0) # Horizontal
        
        # Viewport Area (Grid + Scene + Gizmo)
        # Using Overlay/Stack layout via default Column? 
        # No, we want them stacked.
        # Fantasmagorie Column stacks vertically.
        # To stack on top of each other (Z-ordering), we need a Container with absolute positioning or Overlay.
        # For this demo, let's put them in sequence to show they exist, or use a "Stack" if available.
        # We don't have explicit Z-stack widget exposed nicely yet except maybe putting Gizmo inside Scene?
        # Actually Grid is usually background.
        # Let's just put Grid, then Scene3D, then Gizmo in a vertical stack for now (not ideal, but proves they work).
        # OR: Put them in a container that fills space.
        
        fanta.Row() # Viewport Content
        fanta.FlexGrow(1.0)
        
        # Grid Background
        fanta.Grid()
        fanta.FlexGrow(1.0)
        
        # Note: In a real app we would render Scene3D ON TOP of Grid.
        # Here we place them side-by-side or use alpha blending?
        # Let's demo Scene3D separately in same row.
        
        fanta.Scene3D()
        fanta.FlexGrow(1.0)
        fanta.CameraPos(0.0, 2.0, 5.0)
        fanta.CameraTarget(0.0, 0.0, 0.0)
        
        # Gizmo (Overlay simulated or just placed)
        # Gizmo usually draws last.
        fanta.TransformGizmo()
        fanta.FlexGrow(1.0) # It will take space.
        fanta.Snap(self.snap_enabled)
        
        fanta.End() # Viewport Row
        
        fanta.End() # Main Column
        fanta.End() # Main Row
        
        fanta.End() # Window

app = DemoApp()

def frame(w, h):
    app.build()

fanta.run(frame)
