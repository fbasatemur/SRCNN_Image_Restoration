# -*- coding: utf-8 -*-


if __name__ == '__main__':
      import sys
      from restoration_GUI import Window
      from PyQt5.QtWidgets import QApplication
      
      app = QApplication(sys.argv)
    
      windowObject = Window()
      
      
      sys.exit(app.exec_())