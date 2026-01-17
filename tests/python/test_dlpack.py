import sys
import os

# Ensure fanta_rust is importable
# Assuming fanta_rust.so is in the current directory or python path
try:
    import fanta_rust
except ImportError:
    # Try adding target/debug to path? Or assume user copied it.
    # The previous tests relied on cp command.
    pass

import unittest

class TestDLPack(unittest.TestCase):
    def test_dlpack_export(self):
        try:
            import torch
        except ImportError:
            print("Skipping DLPack test: torch not installed")
            return

        print("Testing DLPack with PyTorch...")
        
        # We need a window to have GL context
        # We define a callback that creates BufferView and tests it
        
        has_run = False
        
        def on_frame(ctx):
            nonlocal has_run
            if has_run:
                return
            
            try:
                print("Inside on_frame, creating BufferView...")
                size = 1024
                bv = fanta_rust.BufferView(size)
                
                print("Mapping...")
                # Map returns a memoryview (PyObject), but we ignore it here and test DLPack
                _ = bv.map()
                
                print("Converting to torch via DLPack...")
                # torch.from_dlpack calls __dlpack__
                t = torch.from_dlpack(bv)
                
                print(f"Tensor created: {t}")
                assert t.shape == (1024,)
                assert t.dtype == torch.uint8
                assert t.device.type == 'cpu'
                
                # Write to tensor (should write to PBO)
                t[0] = 123
                t[1023] = 42
                
                # Check memoryview reflects change (optional, if we kept it)
                # print("DLPack test passed inside frame.")
                
                # Unmap (optional, but good practice)
                # bv.unmap() 
                # Note: if we unmap, tensor 't' points to invalid memory?
                # For this test, valid.
                
            except Exception as e:
                print(f"DLPack Test FAILED: {e}")
                # We can't easily fail the outer test from callback?
                # We'll print and exit?
                sys.exit(1)
            
            has_run = True
            print("DLPack Test SUCCESS")
            sys.exit(0) # Exit success
            
        print("Starting Window...")
        # Run for 1 frame effectively
        fanta_rust.run_window("DLPack Test", 100, 100, on_frame)

if __name__ == '__main__':
    unittest.main()
