# mnist-specialist-generalist

make python notebook
using dataset multimnist handwritten digit like in https://github.com/shaohua0116/MultiDigitMNIST with many small digit as multi task learning 
use mnist handwritten digit as specialist with give padding so the size digit almost the same with multimnist for specialist resnet18
use multimnist for learning generalist resnet50
context semisupervised so use soft labeling from specialist and make it the label for generalist like in transfer learning and knowledge distillation
use pareto exploration and pareto  expand to specialist and generalist pcgrad or mgda or famo

https://www.codabench.org/competitions/18282/?secret_key=98709548-de7e-467d-85db-5d561397fa23#/pages-tab
