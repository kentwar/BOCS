import os
import javabridge

import sys
print(sys.version)

javabridge.start_vm(run_headless=True)
try:
    print javabridge.run_script('java.lang.String.format("Hello, %s!", greetee);',
                                dict(greetee='world'))
finally:
    javabridge.kill_vm()
