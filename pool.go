package n2

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
)

type MaxPoolLayer struct {
	dims       intV2
	inputCache []*mat.Dense
}

func NewMaxPool(size int) (l *MaxPoolLayer) {
	l = &MaxPoolLayer{
		dims: intV2{
			x: size,
			y: size,
		},
	}

	return l
}

func (l *MaxPoolLayer) Forward(input []*mat.Dense) []*mat.Dense {
	out := make([]*mat.Dense, len(input))

	copyT3(&l.inputCache, &input)

	for d := 0; d < len(input); d++ {
		r, c := input[d].Dims()
		out[d] = mat.NewDense(r/l.dims.y, c/l.dims.x, nil)
		for i := 0; i < r/l.dims.y; i++ {
			for j := 0; j < c/l.dims.x; j++ {
				out[d].Set(i, j, mat.Max(input[d].Slice(i*l.dims.y, (i+1)*l.dims.y, j*l.dims.x, (j+1)*l.dims.x)))
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
		for i := 0; i < r/l.dims.y; i++ {
			for j := 0; j < c/l.dims.x; j++ {
				sel := mat.Max(l.inputCache[d].Slice(i*l.dims.y, (i+1)*l.dims.y, j*l.dims.x, (j+1)*l.dims.x))

				for io := 0; io < l.dims.y; io++ {
					for jo := 0; jo < l.dims.x; jo++ {
						if l.inputCache[d].At(i*l.dims.y+io, j*l.dims.x+jo) == sel {
							inputGradient[d].Set(i*l.dims.y+io, j*l.dims.x+jo, outputGradient[d].At(i, j))
						}
					}
				}
			}
		}
	}

	// Phew, That was painful!
	return inputGradient
}

type MaxPoolLayerJSON struct {
	LType  string
	Dx, Dy int
}

func (l *MaxPoolLayer) MarshalJSON() ([]byte, error) {
	lj := &MaxPoolLayerJSON{
		LType: "MaxPool",
		Dx:    l.dims.x,
		Dy:    l.dims.y,
	}

	return json.Marshal(lj)
}

func (l *MaxPoolLayer) UnmarshalJSON(bytes []byte) error {
	lj := &MaxPoolLayerJSON{}

	var err error
	err = json.Unmarshal(bytes, lj)
	if err != nil {
		return err
	}

	l.dims = intV2{
		x: lj.Dx,
		y: lj.Dy,
	}

	return nil
}
