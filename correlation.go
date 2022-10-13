package n2

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func correlateValid(input *mat.Dense, kernel *mat.Dense) *mat.Dense {
	kr, kc := kernel.Dims()
	r, c := input.Dims()

	var (
		out  = mat.NewDense(1+r-kr, 1+c-kc, nil)
		temp = mat.NewDense(kr, kc, nil)
	)

	for i := 0; i < 1+r-kr; i++ {
		for j := 0; j < 1+c-kc; j++ {
			temp.MulElem(kernel, input.Slice(i, i+kr, j, j+kc))
			out.Set(i, j, mat.Sum(temp))
		}
	}

	return out
}

func convolveFull(input *mat.Dense, kernel *mat.Dense) *mat.Dense {
	kr, kc := kernel.Dims()

	var (
		temp          = mat.NewDense(kr, kc, nil)
		inputPadded   = padMat(input, kr+kc-2)
		kernelRotated = rotateMat180(kernel)
	)

	ir, ic := inputPadded.Dims()
	out := mat.NewDense(1+ir-kr, 1+ic-kc, nil)

	for i := 0; i < 1+ir-kr; i++ {
		for j := 0; j < 1+ic-kc; j++ {
			temp.MulElem(kernelRotated, inputPadded.Slice(i, i+kr, j, j+kc))
			out.Set(i, j, mat.Sum(temp))
		}
	}

	return out
}

type CorrLayer struct {
	depth       int
	inputDepth  int
	inputShape  intV3
	outputShape intV3
	kernelShape intV4

	kernels [][]*mat.Dense
	biases  []*mat.Dense

	inputCache  []*mat.Dense
	outputCache []*mat.Dense
}

func NewCorr(inputShape [3]int, kernelSize int, depth int) (l *CorrLayer) {
	l = &CorrLayer{
		depth:      depth,
		inputDepth: inputShape[2],
		inputShape: intV3{
			x: inputShape[0],
			y: inputShape[1],
			z: inputShape[2],
		},
		outputShape: intV3{
			x: 1 + inputShape[0] - kernelSize,
			y: 1 + inputShape[1] - kernelSize,
			z: depth,
		},
		kernelShape: intV4{
			x: kernelSize,
			y: kernelSize,
			z: inputShape[2],
			w: depth,
		},
	}

	l.kernels = makeT4d(l.kernelShape, func(n int) float64 {
		dist := distuv.Uniform{
			Min: -1 / float64(n),
			Max: 1 / float64(n),
		}

		return dist.Rand()
	})
	l.biases = makeT3d(l.outputShape, func(n int) float64 {
		dist := distuv.Uniform{
			Min: -1 / float64(n),
			Max: 1 / float64(n),
		}

		return dist.Rand()
	})

	return l
}

func (l *CorrLayer) Forward(input []*mat.Dense) []*mat.Dense {
	copyT3(&l.inputCache, &input)
	copyT3(&l.outputCache, &l.biases)

	for i := 0; i < l.depth; i++ {
		for j := 0; j < l.inputDepth; j++ {
			c := correlateValid(l.inputCache[j], l.kernels[i][j])

			l.outputCache[i].Add(l.outputCache[i], c)
		}
	}

	return l.outputCache
}

func (l *CorrLayer) Backward(outputGradient []*mat.Dense, learnRate float64) (inputGradient []*mat.Dense) {
	kernelGradient := makeT4d(l.kernelShape, func(n int) float64 {
		return 0
	})

	inputGradient = makeT3d(l.inputShape, func(n int) float64 {
		return 0
	})

	for i := 0; i < l.depth; i++ {
		for j := 0; j < l.inputDepth; j++ {
			kernelGradient[i][j] = correlateValid(l.inputCache[j], outputGradient[i])
			inputGradient[j].Add(inputGradient[j], convolveFull(outputGradient[i], l.kernels[i][j]))
		}
	}

	var (
		kgs = mat.NewDense(l.kernelShape.y, l.kernelShape.x, nil)
		ogs = mat.NewDense(l.outputShape.y, l.outputShape.x, nil)
	)

	for i := 0; i < l.depth; i++ {
		for j := 0; j < l.inputDepth; j++ {
			kgs.Scale(learnRate, kernelGradient[i][j])
			l.kernels[i][j].Sub(l.kernels[i][j], kgs)
		}

		ogs.Scale(learnRate, outputGradient[i])
		l.biases[i].Sub(l.biases[i], ogs)
	}

	return inputGradient
}

type CorrLayerJSON struct {
	LType          string
	Depth          int
	InputDepth     int
	Ix, Iy, Iz     int
	Ox, Oy, Oz     int
	Kx, Ky, Kz, Kw int

	Kernels [][][]byte
	Biases  [][]byte
}

func (l *CorrLayer) MarshalJSON() ([]byte, error) {
	lj := &CorrLayerJSON{
		LType:      "Corr",
		Depth:      l.depth,
		InputDepth: l.inputDepth,
		Ix:         l.inputShape.x,
		Iy:         l.inputShape.y,
		Iz:         l.inputShape.z,
		Ox:         l.outputShape.x,
		Oy:         l.outputShape.y,
		Oz:         l.outputShape.z,
		Kx:         l.kernelShape.x,
		Ky:         l.kernelShape.y,
		Kz:         l.kernelShape.z,
		Kw:         l.kernelShape.w,
		Kernels:    make([][][]byte, len(l.kernels)),
		Biases:     make([][]byte, len(l.biases)),
	}

	var err error
	for i := 0; i < l.depth; i++ {
		lj.Kernels[i] = make([][]byte, len(l.kernels[i]))

		lj.Biases[i], err = l.biases[i].MarshalBinary()
		if err != nil {
			return nil, err
		}

		for j := 0; j < l.inputDepth; j++ {
			lj.Kernels[i][j], err = l.kernels[i][j].MarshalBinary()
			if err != nil {
				return nil, err
			}
		}
	}

	return json.Marshal(lj)
}

func (l *CorrLayer) UnmarshalJSON(bytes []byte) error {
	lj := &CorrLayerJSON{}

	var err error
	err = json.Unmarshal(bytes, lj)

	*l = CorrLayer{
		depth:      lj.Depth,
		inputDepth: lj.InputDepth,
		inputShape: intV3{
			x: lj.Ix,
			y: lj.Iy,
			z: lj.Iz,
		},
		outputShape: intV3{
			x: lj.Ox,
			y: lj.Oy,
			z: lj.Oz,
		},
		kernelShape: intV4{
			x: lj.Kx,
			y: lj.Ky,
			z: lj.Kz,
			w: lj.Kw,
		},
		kernels: make([][]*mat.Dense, len(lj.Kernels)),
		biases:  make([]*mat.Dense, len(lj.Biases)),
	}

	for i := 0; i < l.depth; i++ {
		l.kernels[i] = make([]*mat.Dense, len(lj.Kernels[i]))

		l.biases[i] = new(mat.Dense)
		err = l.biases[i].UnmarshalBinary(lj.Biases[i])
		if err != nil {
			return err
		}

		for j := 0; j < l.inputDepth; j++ {
			l.kernels[i][j] = new(mat.Dense)
			err = l.kernels[i][j].UnmarshalBinary(lj.Kernels[i][j])
			if err != nil {
				return err
			}
		}
	}

	return nil
}
