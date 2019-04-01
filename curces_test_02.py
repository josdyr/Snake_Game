import curses

myscreen = curses.initscr()
curses.start_color()
curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
y,x = myscreen.getmaxyx()
myscreen.addstr("Python curses in action!", curses.color_pair(1))
myscreen.move(y -1,0)
myscreen.addstr("Python curses in action!", curses.color_pair(1))
myscreen.refresh()
myscreen.getch()

curses.endwin()
