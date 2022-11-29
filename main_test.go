package n2

import (
	_ "embed"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
}

func TestNetwork_ToGob(t *testing.T) {
	n := NewNetwork(
		NewCorr([3]int{28, 28, 1}, 5, 3),
		NewCorr([3]int{24, 24, 1}, 3, 3),
		NewFlatten(),
		NewDense(22*22*3, 50),
		NewActSigmoid(),
	)

	n.LearnRate = 0.1

	bytes, err := n.ToGob()
	if err != nil {
		t.Fatal(err)
	}

	on := Network{}
	err = on.FromGob(bytes)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTrain_XOR(t *testing.T) {
	n := NewNetwork(
		NewDense(2, 5),
		NewActSigmoid(),
		NewDense(5, 5),
		NewActSigmoid(),
		NewDense(5, 1),
		NewActSigmoid(),
	)

	n.LearnRate = 0.01

	avgCost := 0.0

	initialCost := -1.0
	endCost := -1.0

	epoch := 100000
	t.Logf("Epoch size of %d", epoch)

	for it := 0; it < 1000000; it++ {
		if it%epoch == 0 && it != 0 {
			avgCost /= float64(epoch)
			if it == epoch {
				initialCost = avgCost
			}
			endCost = avgCost
			t.Logf("Average epoch cost: %f", avgCost)
			avgCost = 0.0
		}

		a, b := rand.Float64(), rand.Float64()
		c := 0.01
		if (a > 0.5 || b > 0.5) && !(a > 0.5 && b > 0.5) {
			c = 0.99
		}

		i := []*mat.Dense{mat.NewDense(2, 1, []float64{a, b})}
		e := []*mat.Dense{mat.NewDense(1, 1, []float64{c})}

		cost, _ := n.Backpropagate(i, e)
		avgCost += cost

		if math.IsNaN(cost) {
			t.Log(a, b, c, cost)
			t.Fail()
		}
	}

	if initialCost < endCost {
		t.Fail()
	}
}
