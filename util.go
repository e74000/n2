package n2

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type intV2 struct {
	x, y int
}

type intV3 struct {
	x, y, z int
}

type intV4 struct {
	x, y, z, w int
}

func wrap(f func(float64) float64) func(i, j int, v float64) float64 {
	return func(i, j int, v float64) float64 {
		return f(v)
	}
}

func randArr(n int, value func(n int) float64) []float64 {
	arr := make([]float64, n)

	for i := 0; i < n; i++ {
		arr[i] = value(n)
	}

	return arr
}

func padMat(input *mat.Dense, padding int) *mat.Dense {
	r, c := input.Dims()
	padded := mat.NewDense(r+padding, c+padding, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			padded.Set(i+padding/2, j+padding/2, input.At(i, j))
		}
	}

	return padded
}

func rotateMat180(input *mat.Dense) *mat.Dense {
	r, c := input.Dims()

	var (
		out = mat.NewDense(r, c, nil)
	)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, c-j-1, input.At(r-i-1, j))
		}
	}

	return out
}

func makeT4d(shape intV4, value func(n int) float64) [][]*mat.Dense {
	out := make([][]*mat.Dense, shape.w)

	for i := 0; i < shape.w; i++ {
		out[i] = make([]*mat.Dense, shape.z)
		for j := 0; j < shape.z; j++ {
			out[i][j] = mat.NewDense(shape.y, shape.x, randArr(shape.y*shape.x, value))
		}
	}

	return out
}

func makeT3d(shape intV3, value func(n int) float64) []*mat.Dense {
	out := make([]*mat.Dense, shape.z)

	for i := 0; i < shape.z; i++ {
		out[i] = mat.NewDense(shape.y, shape.x, randArr(shape.y*shape.x, value))
	}

	return out
}

func reshapeT3toT1(input []*mat.Dense) []*mat.Dense {
	r, c := input[0].Dims()
	out := make([]*mat.Dense, 1)

	out[0] = mat.NewDense(r*c*len(input), 1, nil)

	for d := 0; d < len(input); d++ {
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				out[0].Set(d*r*c+i*c+j, 0, input[d].At(i, j))
			}
		}
	}

	return out
}

func reshapeT1toT3(input []*mat.Dense, shape intV3) []*mat.Dense {
	ir, ic := input[0].Dims()
	if ir*ic != shape.x*shape.y*shape.z {
		panic("Shape doesn't match input!")
	}

	out := make([]*mat.Dense, shape.z)

	for d := 0; d < shape.z; d++ {
		out[d] = mat.NewDense(shape.y, shape.x, nil)
		for i := 0; i < shape.y; i++ {
			for j := 0; j < shape.x; j++ {
				out[d].Set(i, j, input[0].At(d*shape.y*shape.x+i*shape.x+j, 0))
			}
		}
	}

	return out
}

func copyT3(dst, src *[]*mat.Dense) {
	*dst = make([]*mat.Dense, len(*src))

	for i, dense := range *src {
		r, c := dense.Dims()
		(*dst)[i] = mat.NewDense(r, c, nil)
		(*dst)[i].Copy(dense)
	}
}

func mSqErr(output, label []*mat.Dense) float64 {
	var temp *mat.Dense

	count := 0
	total := 0.0

	for d := 0; d < len(label); d++ {
		r, c := label[d].Dims()
		temp = mat.NewDense(r, c, nil)
		temp.Sub(label[d], output[d])
		temp.Apply(wrap(func(v float64) float64 {
			return v * v
		}), temp)
		count += r * c
		total += mat.Sum(temp)
	}

	return total / float64(count)
}

func mSqErrPrime(output, label []*mat.Dense) []*mat.Dense {
	r, c := label[0].Dims()
	n := float64(len(label) * r * c)

	out := make([]*mat.Dense, len(label))

	for d := 0; d < len(label); d++ {
		out[d] = mat.NewDense(r, c, nil)
		out[d].Sub(label[d], output[d])
		out[d].Scale(2/n, out[d])
	}

	return out
}

func closeEnough(outputs, label []*mat.Dense) float64 {
	var total float64
	for i, l := range label {
		r, c := l.Dims()
		temp := mat.NewDense(r, c, nil)

		temp.Sub(l, outputs[i])
		temp.Apply(wrap(math.Abs), temp)

		total += mat.Sum(temp)
	}

	return total
}
