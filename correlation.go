package n2

import (
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
	Depth       int
	InputDepth  int
	InputShape  intV3
	OutputShape intV3
	KernelShape intV4

	Kernels [][]*mat.Dense
	Biases  []*mat.Dense

	inputCache  []*mat.Dense
	outputCache []*mat.Dense
}

func NewCorr(inputShape [3]int, kernelSize int, depth int) (l *CorrLayer) {
	l = &CorrLayer{
		Depth:      depth,
		InputDepth: inputShape[2],
		InputShape: intV3{
			X: inputShape[0],
			Y: inputShape[1],
			Z: inputShape[2],
		},
		OutputShape: intV3{
			X: 1 + inputShape[0] - kernelSize,
			Y: 1 + inputShape[1] - kernelSize,
			Z: depth,
		},
		KernelShape: intV4{
			X: kernelSize,
			Y: kernelSize,
			Z: inputShape[2],
			W: depth,
		},
	}

	l.Kernels = makeT4d(l.KernelShape, func(n int) float64 {
		dist := distuv.Uniform{
			Min: -1 / float64(n),
			Max: 1 / float64(n),
		}

		return dist.Rand()
	})
	l.Biases = makeT3d(l.OutputShape, func(n int) float64 {
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
	copyT3(&l.outputCache, &l.Biases)

	for i := 0; i < l.Depth; i++ {
		for j := 0; j < l.InputDepth; j++ {
			c := correlateValid(l.inputCache[j], l.Kernels[i][j])
			l.outputCache[i].Add(l.outputCache[i], c)
		}
	}

	return l.outputCache
}

// Backward There is something very wrong with the maths for this method
func (l *CorrLayer) Backward(outputGradient []*mat.Dense, learnRate float64) (inputGradient []*mat.Dense) {
	kernelGradient := makeT4d(l.KernelShape, func(n int) float64 {
		return 0
	})

	inputGradient = makeT3d(l.InputShape, func(n int) float64 {
		return 0
	})

	for i := 0; i < l.Depth; i++ {
		for j := 0; j < l.InputDepth; j++ {
			kernelGradient[i][j] = correlateValid(l.inputCache[j], outputGradient[i])
			inputGradient[j].Add(inputGradient[j], convolveFull(outputGradient[i], l.Kernels[i][j]))
		}
	}

	var (
		kgs = mat.NewDense(l.KernelShape.Y, l.KernelShape.X, nil)
		ogs = mat.NewDense(l.OutputShape.Y, l.OutputShape.X, nil)
	)

	for i := 0; i < l.Depth; i++ {
		for j := 0; j < l.InputDepth; j++ {
			kgs.Scale(learnRate, kernelGradient[i][j])
			l.Kernels[i][j].Add(l.Kernels[i][j], kgs)
		}

		ogs.Scale(learnRate, outputGradient[i])
		l.Biases[i].Add.Biases[i], ogs)
	}

	return inputGradient
}
