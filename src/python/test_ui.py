import fanta_rust as fanta

def render(width, height):
    w = float(width)
    h = float(height)

    # Root
    fanta.Column().size(w, h).bg(fanta.Color(0.1, 0.1, 0.12, 1.0)).padding(20)

    fanta.Text("Phase 13: Widgets").font_size(32).color(fanta.Color(1.0, 1.0, 1.0, 1.0))
    fanta.Box().height(20)

    # --- Slider ---
    fanta.Text("Slider").font_size(18).color(fanta.Color(0.8, 0.8, 0.8, 1.0))
    # Note: State management usually happens in Python class or global state.
    # Here we simulate stateless immediate mode with local var (which resets every frame in this script scope unless global).
    # In real app, 'val' comes from persistent state.
    val = 0.5 
    # Use fanta.Slider(val, min, max)
    # Returns builder with updated value? No, builder has get_value().
    slider = fanta.Slider(val, 0.0, 1.0).width(200).height(30)
    new_val = slider.get_value()
    # Logic to print if changed
    # In this stateless test, it will always be initial val + delta of ONE frame. 
    # To test properly we need state. But hot_reload_windowed invokes render() repeatedly.
    # We can't store state easily here in function scope.
    
    fanta.Box().height(20)

    # --- Toggle ---
    fanta.Text("Toggle").font_size(18)
    toggle = fanta.Toggle(False).height(30)
    
    fanta.Box().height(20)

    # --- TextInput ---
    fanta.Text("Text Input").font_size(18)
    
    # Simulate persistent state using a global cache
    if not hasattr(render, "text_state"):
        render.text_state = "Type here..."
        
    text_input = fanta.TextInput(render.text_state).width(200).height(30)
    render.text_state = text_input.get_value()
    
    fanta.Text(f"You typed: {render.text_state}").font_size(14).color(fanta.Color(0.6, 0.6, 0.6, 1.0))
    
    fanta.Box().height(20)

    # --- Clipping / Scroll View ---
    fanta.Text("Clipping (Scroll View)").font_size(18)
    
    # container with clip=True
    fanta.Column().size(300, 100).bg(fanta.Color(0.2, 0.2, 0.22, 1.0)).clip(True).radius(10)
    
    # Content larger than container
    fanta.Text("Content Line 1").font_size(24)
    fanta.Text("Content Line 2").font_size(24)
    fanta.Text("Content Line 3 (Should be clipped)").font_size(24)
    fanta.Text("Content Line 4 (Hidden)").font_size(24)
    
    fanta.End() # End Clip Container

    # --- Image & Bezier ---
    fanta.Text("Visual Arts: Bezier & Image").font_size(18)
    
    # Bezier
    fanta.Bezier(
        (350.0, 500.0), (400.0, 400.0), (500.0, 600.0), (550.0, 500.0)
    ).thickness(3.0).color(fanta.Color(1.0, 0.5, 0.0, 1.0))
    
    # Image (Placeholder path)
    fanta.Image("assets/logo.png").size(100, 100)
    
    # Splitter Demo
    if not hasattr(render, "split_ratio"):
        render.split_ratio = 0.5

    fanta.Text("Splitter").font_size(18)
    
    # Splitter container
    s = fanta.Splitter(render.split_ratio).is_vertical(False).ratio(render.split_ratio)
    
    # Left
    fanta.Box().bg(fanta.Color(0.3, 0.2, 0.2, 1.0)).height(50)
    
    # Right
    fanta.Box().bg(fanta.Color(0.2, 0.2, 0.3, 1.0)).height(50)
    
    fanta.End() # End Splitter
    
    # Update ratio from logic (if implemented)
    render.split_ratio = s.get_ratio()

    fanta.Box().height(20)

    # ColorPicker Demo
    if not hasattr(render, "color_hsv"):
        render.color_hsv = (0.0, 1.0, 1.0)
    
    fanta.Text("Color Picker").font_size(18)
    
    h, s, v = render.color_hsv
    cp = fanta.ColorPicker(h, s, v)
    render.color_hsv = (cp.get_h(), cp.get_s(), cp.get_v())
    
    # Show picked color
    fanta.Box().size(50, 50).bg(cp.get_color())
    fanta.Text(f"H:{h:.2f} S:{s:.2f} V:{v:.2f}").font_size(14)


    with fanta.Column().padding(20).gap(20):
        # Icon demo (using characters that might map to icons in some fonts)
        fanta.Button("Settings").icon("⚙").radius(10).bg(fanta.Color(0.2, 0.2, 0.3, 1))
        fanta.Text("Refresh").icon("↻").font_size(20).fg(fanta.Color(0.4, 0.8, 0.6, 1))
        
        # Per-corner radius demo
        fanta.Box().size(100, 100).bg(fanta.Color(0.2, 0.5, 0.8, 1)).radius_tl(20).radius_br(20)
        
        # Border demo
        fanta.Box().size(100, 100).bg(fanta.Color(0.1, 0.1, 0.1, 0.5)).radius(15).border(2, fanta.Color(1, 1, 0, 1))

        # Nested with rounded image (if we had Image test)
        # fanta.Image("logo.png").size(64, 64).radius(32)

    # Scroll Demo
    fanta.Scroll().size(400, 200).bg(fanta.Color(0.1, 0.1, 0.1, 1.0))
    fanta.Column()
    for i in range(20):
        c = (i / 20.0)
        fanta.Text(f"Scroll Item {i}").color(fanta.Color(c, 1.0-c, 1.0, 1.0))
    fanta.End()
    fanta.End() # End Scroll

    # --- Blur Demo (Glassmorphism) ---
    fanta.Column().size(300, 150).radius(20).blur(10.0).bg(fanta.Color(1.0, 1.0, 1.0, 0.1)).padding(20)
    fanta.Text("Frosted Glass").font_size(24).color(fanta.Color(1.0, 1.0, 1.0, 1.0))
    fanta.Text("Backdrop blur is active").font_size(14).color(fanta.Color(0.8, 0.8, 0.8, 0.8))
    # --- Markdown Demo ---
    fanta.Box().height(20)
    fanta.Text("Markdown Widget").font_size(18)
    
    md_text = """# Hello Markdown
This is a **Fantasmagorie** markdown widget.
## Subheader
- List items (not implemented yet, but text works)
- More text
---
Horizontal rules? maybe.
    """
    # --- Path Demo ---
    fanta.Box().height(20)
    fanta.Text("Adaptive Path Rendering").font_size(18)
    
    p = fanta.Path()
    p.move_to(100, 700)
    p.cubic_to(200, 600, 300, 800, 400, 700)
    p.line_to(400, 800)
    p.line_to(100, 800)
    p.close()
    fanta.DrawPath(p).color(fanta.Color(0.2, 0.6, 1.0, 1.0)).thickness(2.0)
    
    fanta.DrawPath(fanta.Path.circle(500, 700, 40).add_polygon(600, 700, 40, 5)).color(fanta.Color.green())

    # --- Animation Demo ---
    fanta.Box().height(20)
    fanta.Text("Advanced Animation").font_size(18)
    
    # Hover effect with ElasticOut
    btn = fanta.Button("Elastic Hover").size(150, 40)
    # Use longer duration for elastic effect to be visible
    w = btn.animate("width", 200 if btn.hovered() else 150, duration=0.8, easing=fanta.Easing.ElasticOut)
    btn.width(w)
    
    # Spring effect
    fanta.Box().height(10)
    box1 = fanta.Box().size(50, 50).bg(fanta.Color.red())
    y_off = box1.animate("y_spring", 50 if btn.hovered() else 0, duration=0.0, easing=fanta.Easing.Spring)
    box1.margin(y_off)




    # --- I18n Demo ---
    fanta.Box().height(20)
    fanta.Text("Localization").font_size(18)
    
    fanta.AddTranslation("jp", "Hello", "こんにちは (Japanese)")
    fanta.AddTranslation("fr", "Hello", "Bonjour (French)")
    
    # Simple Toggle for language
    if not hasattr(render, "lang"):
        render.lang = "en"
        
    fanta.SetLocale(render.lang)
    fanta.Text(fanta.t("Hello")).font_size(24)
    
    if fanta.Button(f"Switch to JP").clicked(): render.lang = "jp"
    if fanta.Button(f"Switch to FR").clicked(): render.lang = "fr"
    if fanta.Button(f"Switch to EN").clicked(): render.lang = "en"

    # --- IME & Screenshot & Icons --- (Phase 15)
    fanta.Box().height(40)
    fanta.Text("IME / Screenshot / Icon Fallback").font_size(24).color(fanta.Color.white())
    
    # 1. Screenshot Button
    if fanta.Button("📷 Capture Screenshot (screenshot.png)").clicked():
        fanta.capture_frame("screenshot.png")
        # In actual engine, this sets a flag and the next frame is saved.

    fanta.Box().height(10)

    # 2. Icon Font Demo (Fallback)
    with fanta.Row().gap(20).padding(10).bg(fanta.Color(0,0,0,0.3)).radius(10):
        # These icons should work via Segoe MDL2 fallback on Windows
        fanta.Text("\ue713").font_size(32).color(fanta.Color(0.4, 0.7, 1.0, 1.0)) # Settings
        fanta.Text("\ue767").font_size(32).color(fanta.Color(1.0, 0.4, 0.4, 1.0)) # Volume
        fanta.Text("\ue706").font_size(32).color(fanta.Color(0.4, 1.0, 0.4, 1.0)) # Cloud
        fanta.Text("Mixed: Text & \ue713").font_size(20)

    fanta.Box().height(10)

    # 3. IME Support (TextInput)
    fanta.Text("IME Input Test (Try entering Japanese/Chinese):").font_size(16)
    if not hasattr(render, "ime_text"):
        render.ime_text = "日本語入力をお試しください"
    
    ti = fanta.TextInput(render.ime_text).width(500).height(40).font_size(24).radius(8).bg(fanta.Color(0.15, 0.15, 0.2, 1.0))
    render.ime_text = ti.get_value()

    fanta.End() # End Root
