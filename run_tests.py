"""run_tests.py - execute every module self-test in order."""
import runpy, sys
mods = ["braille_core","display_driver","input_handler","ai_backend","cloud_sync","device"]
for m in mods:
    print(f"\n========== {m} ==========")
    runpy.run_module(m, run_name="__main__")
print("\nALL MODULE SELF-TESTS COMPLETED OK")