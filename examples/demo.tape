# Terminal recording for the README, made with https://github.com/charmbracelet/vhs
#   vhs examples/demo.tape   ->   docs/assets/demo.gif
Output docs/assets/demo.gif
Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Env CALLM_HOME "/tmp/callm-demo"
Hide
Type "rm -rf /tmp/callm-demo && clear" Enter
Show

Type "python examples/offline_demo.py" Sleep 500ms Enter
Sleep 4s
Type "callm stats" Sleep 500ms Enter
Sleep 4s
Type "callm calls --limit 5" Sleep 500ms Enter
Sleep 5s
