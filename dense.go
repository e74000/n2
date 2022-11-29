package n2

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type DenseLayer struct {
	Weights *mat.Dense
	Biases  *mat.Dense

	inputCache  []*mat.Dense
	outputCache []*mat.Dense
}

func NewDense(inputs, outputs int) (l *DenseLayer) {
	l = &DenseLayer{
		Weights: mat.NewDense(outputs, inputs, randArr(outputs*inputs, func(n int) float64 {
			dist := distuv.Uniform{
				Min: -1,
				Max: 1,
			}

			return dist.Rand()
		})),
		Biases: mat.NewDense(outputs, 1, randArr(outputs, func(n int) float64 {
			dist := distuv.Uniform{
				Min: -1,
				Max: 1,
			}

			return dist.Rand()
		})),
	}

	return l
}

func (l *DenseLayer) Forward(inputs []*mat.Dense) []*mat.Dense {
	r, c := l.Biases.Dims()
	out := mat.NewDense(r, c, nil)

	out.Product(l.Weights, inputs[0])
	out.Add(out, l.Biases)

	copyT3(&l.inputCache, &inputs)
	copyT3(&l.outputCache, &[]*mat.Dense{out})

	return []*mat.Dense{out}
}

func (l *DenseLayer) Backward(outputGradient []*mat.Dense, learnRate float64) (inputGradient []*mat.Dense) {
	rw, cw := l.Weights.Dims()
	weightsGradient := mat.NewDense(rw, cw, nil)

	weightsGradient.Product(outputGradient[0], l.inputCache[0].T())

	biasGradient := mat.DenseCopyOf(outputGradient[0])

	inputGradient = []*mat.Dense{mat.NewDense(cw, 1, nil)}
	inputGradient[0].Product(l.Weights.T(), outputGradient[0])

	weightsGradient.Scale(learnRate, weightsGradient)
	biasGradient.Scale(learnRate, biasGradient)

	l.Weights.Add(l.Weights, weightsGradient)
	l.Biases.Add(l.Biases, biasGradient)

	return inputGradient
}
