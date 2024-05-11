package engine

func (book *OrderBook) Process(order Order) []Trade {
	if order.Side == 1 {
		return book.ProcessLimitBuy(order)
	}
	return book.ProcessLimitSell(order)
}

func (book *OrderBook) ProcessLimitBuy(order Order) []Trade {
	trades := make([]Trade, 0, 1)
	n := len(book.SellOrders)

	if n == 0 {
		return trades
	}

	if book.SellOrders[n-1].Price <= order.Price {
		// traverse all orders that match
		for i := n - 1; i >= 0; i-- {
			sellOrder := book.SellOrders[i]
			if sellOrder.Price > order.Price {
				break
			}
			//fill the entire order
			if sellOrder.Amount >= order.Amount {
				trades = append(trades, Trade{order.ID, sellOrder.ID, order.Amount, sellOrder.Price})
				sellOrder.Amount -= order.Amount
				if sellOrder.Amount == 0 {
					book.removeSOrder(i)
				}
				return trades
			}
			// fill partial order and continue
			if sellOrder.Amount < order.Amount {
				trades = append(trades, Trade{order.ID, sellOrder.ID, sellOrder.Amount, sellOrder.Price})
				order.Amount -= sellOrder.Amount
				book.removeSOrder(i)
				continue
			}
		}
	}
	// add the remaining order to the list
	book.addBuyOrder(order)
	return trades
}

func (book *OrderBook) ProcessLimitSell(order Order) []Trade {
	trades := make([]Trade, 0, 1)
	n := len(book.BuyOrders)
	if n == 0 {
		return trades
	}
	if book.BuyOrders[n-1].Price >= order.Price {
		// traverse all orders that match
		for i := n - 1; i >= 0; i-- {
			buyOrder := book.BuyOrders[i]
			if buyOrder.Price < order.Price {
				break
			}
			// fill the entire order
			if buyOrder.Amount >= order.Amount {
				trades = append(trades, Trade{order.ID, buyOrder.ID, buyOrder.Amount, buyOrder.Price})
				buyOrder.Amount -= order.Amount
				if buyOrder.Amount == 0 {
					book.removeBOrder(i)
				}
				return trades
			}
			// fill the partial order
			if buyOrder.Amount < order.Amount {
				trades = append(trades, Trade{order.ID, buyOrder.ID, buyOrder.Amount, buyOrder.Price})
				order.Amount -= buyOrder.Amount
				book.removeBOrder(i)
				continue
			}
		}
	}
	// add remaining order to the list
	book.addSellOrder(order)
	return trades
}
