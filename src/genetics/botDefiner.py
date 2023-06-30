class BotDefiner:

  bot:List[Any]
  params:List[Any]
  _index: int
  debug: bool
  def __init__(self, bot = None, debug = False):
      super()
      self.bot = bot
      self.params = []
      self.reset()
      self.debug = debug

  def Boolean(self, name = None):
    return self.Int(0,1, name  = name)
  
  def Int(self, *args , name = None):
    return self.getRand(lambda: random.randint(args[0],args[1]), name = name)

  def getRand(self, func, name = None):
    if (self.debug):
      print("name = ", name)
    if self.bot != None:
      if (self.debug):
        print("current _index:", self._index)
      return_value = self.bot[self._index]
      
      if (self.debug):
       print("current _return value:", return_value)
      self._index +=1
      return return_value
    else:
      self.params.append((func, name))
      return func()


  def reset(self, doClean = False):
    self._index = 0
    if doClean:
      self.params = []

  def loadBot(self, bot):
    self.bot = bot
    self.reset()
    return self

  def generateRandomBot(self):
    bot = []
    if (self.debug):
      print("start generating")
    for f, name in self.params:
        if (self.debug):
          print(f, name)
        bot.append(f())
    if (self.debug):
      print("stop generating")
    return botgi