package n2

import (
	_ "embed"
	"fmt"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
}

func TestNetwork_ToGob(t *testing.T) {
	n := NewNetwork([]Layer{
		NewCorr([3]int{28, 28, 1}, 5, 3),
		NewCorr([3]int{24, 24, 1}, 3, 3),
		NewFlatten(),
		NewDense(22*22*3, 50),
		NewActSigmoid(),
	}, 0.1)

	bytes, err := n.ToGob()
	if err != nil {
		panic(err)
	}

	on := Network{}
	err = on.FromGob(bytes)
	if err != nil {
		panic(err)
	}

	aT, aG := n.Layers[4].(*ActivationLayer).AFunc.Tokenise(), on.Layers[4].(*ActivationLayer).AFunc.Tokenise()
	pT, pG := n.Layers[4].(*ActivationLayer).PFunc.Tokenise(), on.Layers[4].(*ActivationLayer).PFunc.Tokenise()

	fmt.Println(aT.String(), aG.String())
	fmt.Println(pT.String(), pG.String())

	for i := 0; i < 5; i++ {
		fmt.Println(n.Layers[i])
		fmt.Println(on.Layers[i])
		fmt.Println()
	}
}
