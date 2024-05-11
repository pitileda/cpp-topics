package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/IBM/sarama"
)

// TODO SIGUSR1 doesn't work, to investigate
// Sarama client for Kafka config options
var (
	brokers  = ""
	version  = ""
	group    = ""
	topics   = ""
	assignor = ""
	oldest   = true
	verbose  = true
)

func init() {
	flag.StringVar(&brokers, "brokers", "", "Kafka brokers to connect to.")
	flag.StringVar(&group, "group", "", "Kafka consumer group definition.")
	flag.StringVar(&version, "version", sarama.DefaultVersion.String(), "Kafka cluster version.")
	flag.StringVar(&topics, "topics", "", "Kafka topics to be consumed.")
	flag.StringVar(&assignor, "assignor", "range", "Consumer strategy (range, roundrobin, sticky).")
	flag.BoolVar(&oldest, "oldest", true, "Consumer initial offset from oldest.")
	flag.BoolVar(&verbose, "verbose", false, "Sarama logging")
	flag.Parse()

	if len(brokers) == 0 {
		panic("No brokers defined")
	}

	if len(topics) == 0 {
		panic("No topics to be consumed")
	}

	if len(group) == 0 {
		panic("No group to be consumed")
	}
}

func main() {
	keepRunning := true
	log.Println("Start new consumer")
	if verbose {
		sarama.Logger = log.New(os.Stdout, "[sarama] ", log.LstdFlags)
	}

	version, err := sarama.ParseKafkaVersion(version)
	if err != nil {
		log.Panicf("Error parsing Kafka version: %v", err)
	}

	config := sarama.NewConfig()
	config.Version = version

	switch assignor {
	case "sticky":
		config.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{sarama.NewBalanceStrategySticky()}
	case "roundrobin":
		config.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{sarama.NewBalanceStrategyRoundRobin()}
	case "range":
		config.Consumer.Group.Rebalance.GroupStrategies = []sarama.BalanceStrategy{sarama.NewBalanceStrategyRange()}
	default:
		log.Panicf("Unrecognized strategy: %s", assignor)
	}
	consumer := Consumer{
		ready: make(chan bool),
	}

	ctx, cancel := context.WithCancel(context.Background())
	client, err := sarama.NewConsumerGroup(strings.Split(brokers, ","), group, config)
	if err != nil {
		log.Panicf("error creating consumer group")
	}

	consumptionIsPaused := false
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			if err := client.Consume(ctx, strings.Split(topics, ","), &consumer); err != nil {
				if errors.Is(err, sarama.ErrClosedConsumerGroup) {
					return
				}
				log.Panicf("Error from consumer: %v", err)
			}
			if ctx.Err() != nil {
				return
			}
			consumer.ready = make(chan bool)
		}
	}()
	<-consumer.ready // Await till consumer has been setup
	log.Println("Sarama consumer is up and running")

	sigusr1 := make(chan os.Signal, 1)
	signal.Notify(sigusr1, syscall.SIGUSR1)
	sigterm := make(chan os.Signal, 1)
	signal.Notify(sigterm, syscall.SIGTERM, syscall.SIGINT)

	for keepRunning {
		select {
		case <-ctx.Done():
			log.Println("terminating: context has been canceled")
			keepRunning = false
		case <-sigterm:
			log.Println("terminating: via signal")
			keepRunning = false
		case <-sigusr1:
			log.Println("Received SIGUSR1 signal")
			toggleConsupmtionFlow(client, &consumptionIsPaused)
		}
	}
	cancel()
	wg.Wait()
	if err := client.Close(); err != nil {
		log.Panicf("Error closing client: %v", err)
	}
}

type Consumer struct {
	ready chan bool
}

func (consumer *Consumer) Setup(sarama.ConsumerGroupSession) error {
	close(consumer.ready)
	return nil
}

func (consumer *Consumer) Cleanup(sarama.ConsumerGroupSession) error {
	return nil
}

func (consumer *Consumer) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	// this cannot be goroutine, ConsumeClaim itself is goroutine
	for {
		select {
		case message, ok := <-claim.Messages():
			if !ok {
				log.Printf("Message channel was closed")
				return nil
			}
			log.Printf("Message claimed: value: %s, timestamp: %v, topic: %s",
				string(message.Value), message.Timestamp, message.Topic)
			session.MarkMessage(message, "")
		case <-session.Context().Done():
			return nil
		}
	}
}

func toggleConsupmtionFlow(client sarama.ConsumerGroup, isPaused *bool) {
	if *isPaused {
		client.ResumeAll()
		log.Println("Resuming consumption")
	} else {
		client.PauseAll()
		log.Println("Pausing consumption")
	}
	*isPaused = !*isPaused
}
