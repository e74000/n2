package n2

import (
	_ "embed"
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
}

func TestNetwork_UnmarshalJSON(t *testing.T) {
	n := NewNetwork([]Layer{
		NewCorr([3]int{28, 28, 1}, 3, 3),
	}, 0.1)

	empty := mat.NewDense(28, 28, nil)

	t2 := n.Feedforward([]*mat.Dense{empty})

	nb, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}

	on := &Network{}

	err = json.Unmarshal(nb, on)
	if err != nil {
		t.Fatal(err)
	}

	out := on.Feedforward([]*mat.Dense{empty})

	for i := 0; i < len(out); i++ {
		if !mat.Equal(out[i], t2[i]) {
			t.Fail()
		}
	}
}
