
import sys
import math
from ui import *

def main():
    window = Window("Fantasmagorie Feature Test", 1200, 800)
    
    # State
    checkbox_val = False
    radio_val = 1
    toggle_val = True
    dropdown_idx = 0
    text_buffer = "Type here..."
    slider_val = 0.5
    
    items = ["Option A", "Option B", "Option C", "Option D"]
    
    while not window.should_close():
        ui = window.begin_ui()
        
        with ui.row():
            # Panel 1: Core Widgets
            with ui.box().width(300).height(700).bg(0.15, 0.15, 0.15, 1.0).padding(20).column():
                ui.text("Core Widgets").size(24).color(0.9, 0.9, 0.9, 1.0)
                ui.separator().margin(10)
                
                # Checkbox
                new_cb = ui.checkbox("Show Details", checkbox_val)
                checkbox_val = new_cb.value
                
                ui.separator().size(10)
                
                # Toggle
                new_toggle = ui.toggle("Enable Physics", toggle_val)
                toggle_val = new_toggle.value
                
                ui.separator().size(10)
                
                # Radio Group
                ui.text("Select Mode:").size(16)
                if ui.radio("Mode 1", radio_val, 1)[0] == 1: radio_val = 1
                if ui.radio("Mode 2", radio_val, 2)[0] == 2: radio_val = 2
                if ui.radio("Mode 3", radio_val, 3)[0] == 3: radio_val = 3
                
                ui.separator().size(10)
                
                # Dropdown
                ui.text("Dropdown:").size(16)
                dropdown_idx, _ = ui.dropdown("Selection", items, dropdown_idx)
                ui.text(f"Selected: {items[dropdown_idx] if dropdown_idx < len(items) else 'None'}")

            # Panel 2: Grid & Text
            with ui.box().flex_grow(1.0).height(700).padding(20).column():
                ui.text("Grid & Text Input").size(24)
                ui.separator().margin(10)
                
                ui.text("Text Input Test:").size(16)
                text_buffer = ui.text_input(text_buffer).width(400).height(40).font_size(18).build().text
                ui.text(f"Buffer: {text_buffer}").color(0.5, 0.5, 0.5, 1.0)
                
                ui.separator().margin(20)
                
                ui.text("Grid Layout (3 Cols):").size(16)
                with ui.grid(3).gap(10).height(300):
                    for i in range(9):
                        color = (0.3 + (i*0.1)%0.7, 0.3 + (i*0.2)%0.7, 0.4, 1.0)
                        ui.box().bg(*color).height(80).radius(8).center().add(
                            ui.text(f"Item {i+1}").color(1,1,1,1)
                        )
                        
                ui.separator().margin(20)
                
                ui.text("Sliders & Extras").size(16)
                slider_val = ui.slider(slider_val, 0.0, 1.0).label("Volume").build().value
                
                ui.fader(slider_val, 0.0, 1.0).label("Fade").height(150)
                
        window.end_ui()
    
if __name__ == "__main__":
    main()
