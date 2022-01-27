from twisted.internet import reactor
from autobahn.websocket import WebSocketClientFactory, WebSocketClientProtocol, connectWS
  
import cppMethods 
  
class EchoClientProtocol(WebSocketClientProtocol):
  
   def sendHello(self):
      self.sendMessage("Hello, world!")  
  
   def onOpen(self):
      self.sendHello()
  
   def onMessage(self, msg, binary):
      cppMethods.printMessage(msg)
      reactor.callLater(1, self.sendHello)
  
def Connect(addressStr):
    factory = WebSocketClientFactory(addressStr)
    factory.protocol = EchoClientProtocol
    connectWS(factory)
    reactor.run()
