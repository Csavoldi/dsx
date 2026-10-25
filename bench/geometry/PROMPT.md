Create render.py, a Python 3 program using only the standard library, that animates a solid 3D $shape rotating about two axes in the terminal.

Use Unicode braille characters (U+2800–U+28FF), treating each character as a 2×4 dot cell. The visible faces must be filled, with clearly different shading so the solid looks three-dimensional. Nearer faces must hide the surfaces behind them. Choose the rendering algorithm yourself.

Keep the solid centered and comfortably sized: it should occupy a substantial part of the terminal while leaving a clear margin, with no clipping. Adapt to terminal resizing, aim for about 30 frames per second, hide the cursor during animation, and restore it on Ctrl-C.

Your display is tmux pane $pane. Run the demo there:
  tmux send-keys -t $pane 'cd $workspace && python3 render.py' Enter
Inspect it with:
  tmux capture-pane -p -t $pane
If a functional repair requires restarting it, first send:
  tmux send-keys -t $pane C-c
Never run the endless animation in your own tool shell.

Self-validate once: confirm that a recognizable shaded solid is visible, it is neither tiny nor clipped, and it animates without errors. Fix a functional failure if needed. Once those checks pass, leave it running and give a brief completion summary. Do not keep adjusting the appearance or repeatedly capture similar frames.

Work only in this task directory and the named display pane. Do not access other services, agents, panes, credentials, or project directories. Do not install packages.
