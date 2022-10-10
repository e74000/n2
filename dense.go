package n2

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type DenseLayer struct {
	weights *mat.Dense
	biases  *mat.Dense

	inputCache  []*mat.Dense
	outputCache []*mat.Dense
}

func newDense(inputs, outputs int) (l *DenseLayer) {
	l = &DenseLayer{
		weights: mat.NewDense(outputs, inputs, randArr(outputs*inputs, func(n int) float64 {
			dist := distuv.Uniform{
				Min: -1,
				Max: 1,
			}

			return dist.Rand()
		})),
		biases: mat.NewDense(outputs, 1, randArr(outputs, func(n int) float64 {
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
	r, c := l.biases.Dims()
	out := mat.NewDense(r, c, nil)

	out.Product(l.weights, inputs[0])
	out.Add(out, l.biases)

	copyT3(&l.inputCache, &inputs)
	copyT3(&l.outputCache, &[]*mat.Dense{out})

	return []*mat.Dense{out}
}

func (l *DenseLayer) Backward(outputGradient []*mat.Dense, learnRate float64) (inputGradient []*mat.Dense) {
	rw, cw := l.weights.Dims()
	weightsGradient := mat.NewDense(rw, cw, nil)

	weightsGradient.Product(outputGradient[0], l.inputCache[0].T())

	biasGradient := mat.DenseCopyOf(outputGradient[0])

	inputGradient = []*mat.Dense{mat.NewDense(cw, 1, nil)}
	inputGradient[0].Product(l.weights.T(), outputGradient[0])

	weightsGradient.Scale(learnRate, weightsGradient)
	biasGradient.Scale(learnRate, biasGradient)

	l.weights.Sub(l.weights, weightsGradient)
	l.biases.Sub(l.biases, biasGradient)

	return inputGradient
}

type DenseLayerJSON struct {
	LType   string
	Weights []byte
	Biases  []byte
}

func (l *DenseLayer) MarshalJSON() ([]byte, error) {
	wb, err := l.weights.MarshalBinary()
	if err != nil {
		return nil, err
	}

	bb, err := l.biases.MarshalBinary()
	if err != nil {
		return nil, err
	}

	lj := &DenseLayerJSON{
		LType:   "Dense",
		Weights: wb,
		Biases:  bb,
	}

	return json.Marshal(lj)
}

func (l *DenseLayer) UnmarshalJSON(bytes []byte) error {
	lj := &DenseLayerJSON{}

	var err error
	err = json.Unmarshal(bytes, lj)
	if err != nil {
		return err
	}

	l.weights = new(mat.Dense)
	err = l.weights.UnmarshalBinary(lj.Weights)
	if err != nil {
		return err
	}

	l.weights = new(mat.Dense)
	err = l.weights.UnmarshalBinary(lj.Biases)
	if err != nil {
		return err
	}

	return nil
}
