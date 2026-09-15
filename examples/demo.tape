# Terminal recording for the README, made with https://github.com/charmbracelet/vhs
#   PATH="$PWD/.venv/bin:$PATH" vhs examples/demo.tape   ->   docs/assets/demo.gif
Output docs/assets/demo.gif
Set Shell "bash"
Set FontSize 16
Set Width 1400
Set Height 780
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type "export CALLM_HOME=/tmp/callm-demo PS1='$ ' && rm -rf /tmp/callm-demo && clear" Enter
Sleep 500ms
Show

Type "python examples/offline_demo.py" Sleep 400ms Enter
Sleep 5s
Type "callm stats" Sleep 400ms Enter
Sleep 4s
Type "callm calls --limit 5" Sleep 400ms Enter
Sleep 6s
