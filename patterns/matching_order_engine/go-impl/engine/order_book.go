package engine

type OrderBook struct {
	BuyOrders  []Order
	SellOrders []Order
}

func (book *OrderBook) addBuyOrder(order Order) {
	n := len(book.BuyOrders)
	var i int
	for i := n - 1; i >= 0; i-- {
		buyOrder := book.BuyOrders[i]
		if buyOrder.Price < order.Price {
			break
		}
	}

	if i == n-1 {
		book.BuyOrders = append(book.BuyOrders, order)
	} else {
		copy(book.BuyOrders[i+1:], book.BuyOrders[i:])
		book.BuyOrders[i] = order
	}
}

func (book *OrderBook) addSellOrder(order Order) {
	n := len(book.SellOrders)
	var i int
	for i := n - 1; i >= 0; i-- {
		sellOrder := book.SellOrders[i]
		if sellOrder.Price > order.Price {
			break
		}
	}
	if i == n-1 {
		book.SellOrders = append(book.SellOrders, order)
	} else {
		copy(book.SellOrders[i+1:], book.SellOrders[i:])
		book.SellOrders[i] = order
	}
}

func (book *OrderBook) removeBOrder(index int) {
	book.BuyOrders = append(book.BuyOrders[:index], book.BuyOrders[index+1:]...)
}

func (book *OrderBook) removeSOrder(index int) {
	book.SellOrders = append(book.SellOrders[:index], book.SellOrders[index+1:]...)
}
