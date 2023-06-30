def bd_test2(bd):
  for i in range(3):
    param1 = 2 ** bd.Int(0,1, name = "0")
    param2 = bd.getRand(lambda: random.choice(("a", "b")), name = '1')
    if bd.Boolean(name = "2"):
      print(param1, param2)
    
bd.reset(doClean = True)
bd_test2(bd)
print(len(bd.params))
bd.params

bd_test2(bd.loadBot(bd.generateRandomBot()))
print(len(bd.bot))
bd.bot