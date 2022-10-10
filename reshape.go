package n2

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
)

func NewFlatten() (l *FlattenLayer) {
	l = new(FlattenLayer)
	return l
}

type FlattenLayer struct {
	inputShape intV3
}

func (l *FlattenLayer) Forward(input []*mat.Dense) []*mat.Dense {
	r, c := input[0].Dims()
	l.inputShape = intV3{
		x: c,
		y: r,
		z: len(input),
	}
	return reshapeT3toT1(input)
}

func (l *FlattenLayer) Backward(outputGradient []*mat.Dense, _ float64) []*mat.Dense {
	return reshapeT1toT3(outputGradient, l.inputShape)
}

type FlattenLayerJSON struct {
	LType      string
	ix, iy, iz int
}

func (l *FlattenLayer) MarshalJSON() ([]byte, error) {
	lj := &FlattenLayerJSON{
		LType: "Flatten",
		ix:    l.inputShape.x,
		iy:    l.inputShape.y,
		iz:    l.inputShape.z,
	}

	return json.Marshal(lj)
}

func (l *FlattenLayer) UnmarshalJSON(byte []byte) error {
	lj := &FlattenLayerJSON{}

	var err error
	err = json.Unmarshal(byte, lj)
	if err != nil {
		return err
	}

	*l = FlattenLayer{
		inputShape: intV3{
			x: lj.ix,
			y: lj.iy,
			z: lj.iz,
		},
	}

	return nil
}
