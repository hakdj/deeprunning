class Athlete:
    def __init__(self, value='Jane'): # 초기화 펑셩은 무조건 넣어야함
        self.inner_value= value;
        print(self.inner_value)

    def getInnerValue(self):
        return self.inner_value

#athlete =Athlete()
#athlete =Athlete.__init__()
#위와 아래는 똑같음

athlete =Athlete(value='hakdj')

print(athlete.getInnerValue())


class Temp(Athlete):
    def __init__(self):
        super()
