library(tensorflow)

hello<-tf$constant("Hello,tensorflow")
sess<-tf$Session()
print(sess$run(hello))
sess$close()

#노드수
node1<-tf$constant(3.0,tf$float32)
node2<-tf$constant(5.0)
node3<-tf$add(node1,node2)
print(node2)
sess$close()

#placeholder
a<-tf$placeholder(tf$float32)
b<-tf$placeholder(tf$float32)

#모델설정
adder.node<-a+b

#automatically close
with(tf$Session() %as% sess, {
  print(sess$run(adder.node,
                 feed_dict = dict(a = 3, b = 4.5)))
  print(sess$run(adder.node,
                 feed_dict = dict(a = c(1:10), b = c(21:30))))
})
