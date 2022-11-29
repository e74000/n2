package n2

import (
	"gonum.org/v1/gonum/mat"
)

type MaxPoolLayer struct {
	Dims       intV2
	inputCache []*mat.Dense
}

func NewMaxPool(size int) (l *MaxPoolLayer) {
	l = &MaxPoolLayer{
		Dims: intV2{
			X: size,
			Y: size,
		},
	}

	return l
}

func (l *MaxPoolLayer) Forward(input []*mat.Dense) []*mat.Dense {
	out := make([]*mat.Dense, len(input))

	copyT3(&l.inputCache, &input)

	for d := 0; d < len(input); d++ {
		r, c := input[d].Dims()
		out[d] = mat.NewDense(r/l.Dims.Y, c/l.Dims.X, nil)
		for i := 0; i < r/l.Dims.Y; i++ {
			for j := 0; j < c/l.Dims.X; j++ {
				out[d].Set(i, j, mat.Max(input[d].Slice(i*l.Dims.Y, (i+1)*l.Dims.Y, j*l.Dims.X, (j+1)*l.Dims.X)))
			}
		}
	}

	return out
}

func (l *MaxPoolLayer) Backward(outputGradient []*mat.Dense, _ float64) (inputGradient []*mat.Dense) {
	r, c := l.inputCache[0].Dims()

	inputGradient = make([]*mat.Dense, len(l.inputCache))

	// Wow, I sure do like quintuple nested for loops!
	for d := 0; d < len(l.inputCache); d++ {
		inputGradient[d] = mat.NewDense(r, c, nil)
		for i := 0; i < r/l.Dims.Y; i++ {
			for j := 0; j < c/l.Dims.X; j++ {
				sel := mat.Max(l.inputCache[d].Slice(i*l.Dims.Y, (i+1)*l.Dims.Y, j*l.Dims.X, (j+1)*l.Dims.X))

				for io := 0; io < l.Dims.Y; io++ {
					for jo := 0; jo < l.Dims.X; jo++ {
						if l.inputCache[d].At(i*l.Dims.Y+io, j*l.Dims.X+jo) == sel {
							inputGradient[d].Set(i*l.Dims.Y+io, j*l.Dims.X+jo, outputGradient[d].At(i, j))
						}
					}
				}
			}
		}
	}

	// Phew, That was painful!
	return inputGradient
}
