import fanta_rust as fanta
import numpy as np
import time

def render(width, height):
    w = float(width)
    h = float(height)

    # Main Container
    fanta.Column().size(w, h).bg(fanta.Color(0.05, 0.05, 0.07, 1.0)).padding(30).gap(25)

    with fanta.Row().gap(20).width_fill():
        fanta.Text("Fantasmagorie").font_size(36).color(fanta.Color(1, 1, 1, 1))
        fanta.Text("V5 Plotting Engine").font_size(36).color(fanta.Color(0.4, 0.7, 1.0, 1.0))

    fanta.Text("Zero-copy NumPy integration & optimized GPU rendering").font_size(16).color(fanta.Color(0.6, 0.6, 0.7, 1))

    with fanta.Row().gap(30).width_fill().height_fill():
        
        # --- Real-time Function Plot ---
        with fanta.Column().width_fill().gap(15):
            fanta.Text("Real-time Line Plot").font_size(20).color(fanta.Color(0.8, 0.8, 1.0, 1))
            
            t = time.time()
            x = np.linspace(0, 10, 500)
            # Dynamic composite wave
            y = np.sin(x + t * 2.0) * 0.4 + np.cos(x * 2.5 - t * 1.2) * 0.3 + np.sin(x * 5.0 + t * 3.0) * 0.1
            
            fanta.Plot() \
                .title("Interference Pattern") \
                .x_range(0, 10) \
                .y_range(-1, 1) \
                .height(300) \
                .line(y.astype(np.float32), fanta.Color.cyan(), width=3.0) \
                .line((y*0.5).astype(np.float32), fanta.Color.red(), width=1.5) \
                .build()
            
            fanta.Text("• FastLine rendering (Batch)").font_size(14).color(fanta.Color(0.5, 0.5, 0.6, 1))
            fanta.Text("• 500 points @ 60 FPS").font_size(14).color(fanta.Color(0.5, 0.5, 0.6, 1))

        # --- High Density Heatmap ---
        with fanta.Column().width_fill().gap(15):
            fanta.Text("GPU-Accelerated Heatmap").font_size(20).color(fanta.Color(1.0, 0.8, 0.8, 1))
            
            res = 128
            xx, yy = np.meshgrid(np.linspace(-3, 3, res), np.linspace(-3, 3, res))
            r = np.sqrt(xx**2 + yy**2)
            # Ripple effect
            data = np.exp(-r*0.5) * np.sin(r*4.0 - t*3.0)
            
            fanta.Plot() \
                .title("2D Gaussian Ripple") \
                .height(300) \
                .heatmap(data.flatten().astype(np.float32), res, res, colormap="viridis", min=-1, max=1) \
                .build()

            fanta.Text(f"• {res}x{res} Optimized Texture upload").font_size(14).color(fanta.Color(0.6, 0.5, 0.5, 1))
            fanta.Text("• Viridis colormap shader").font_size(14).color(fanta.Color(0.6, 0.5, 0.5, 1))

    fanta.End() # End Root
