"""
Poker Decision Logic

This script implements basic poker strategy and decision-making logic.
It evaluates hand strength, calculates pot odds, and determines optimal actions.

Requirements:
    - treys (poker hand evaluator)
    - Basic poker knowledge

Usage:
    # Evaluate hand strength
    python poker_logic.py --hand "A♠ K♠" --board "Q♠ J♠ 10♠"

    # Get recommended action
    python poker_logic.py --hand "7♥ 2♦" --board "K♠ Q♠ J♣" --pot 100 --bet 50
"""

import argparse
from typing import List, Tuple, Optional
import json


class PokerHandEvaluator:
    """Evaluate poker hand strength and rankings."""

    # Hand rankings (highest to lowest)
    HAND_RANKINGS = {
        'Royal Flush': 10,
        'Straight Flush': 9,
        'Four of a Kind': 8,
        'Full House': 7,
        'Flush': 6,
        'Straight': 5,
        'Three of a Kind': 4,
        'Two Pair': 3,
        'One Pair': 2,
        'High Card': 1
    }

    RANK_VALUES = {
        'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
        '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
        '4': 4, '3': 3, '2': 2
    }

    SUIT_SYMBOLS = {'♠': 'spades', '♥': 'hearts', '♦': 'diamonds', '♣': 'clubs'}

    def __init__(self):
        """Initialize hand evaluator."""
        # Try to load treys library for advanced evaluation
        self.use_treys = False
        try:
            from treys import Card, Evaluator
            self.Card = Card
            self.treys_evaluator = Evaluator()
            self.use_treys = True
            print("✓ Using treys library for hand evaluation")
        except ImportError:
            print("⚠ treys not installed, using basic evaluation")
            print("Install with: pip install treys")

    def parse_card(self, card_str):
        """
        Parse card string to rank and suit.

        Args:
            card_str (str): Card string (e.g., "A♠", "10♥")

        Returns:
            tuple: (rank, suit)
        """
        # Extract rank (handles "10" as two characters)
        if card_str.startswith('10'):
            rank = '10'
            suit = card_str[2]
        else:
            rank = card_str[0]
            suit = card_str[1] if len(card_str) > 1 else ''

        return rank, suit

    def evaluate_hand_basic(self, hole_cards, community_cards):
        """
        Basic hand evaluation without treys library.

        Args:
            hole_cards (list): List of 2 hole cards
            community_cards (list): List of 3-5 community cards

        Returns:
            dict: Hand evaluation result
        """
        all_cards = hole_cards + community_cards

        # Parse all cards
        cards = []
        for card_str in all_cards:
            rank, suit = self.parse_card(card_str)
            cards.append({'rank': rank, 'suit': suit, 'value': self.RANK_VALUES.get(rank, 0)})

        # Sort by value
        cards.sort(key=lambda x: x['value'], reverse=True)

        # Check for flush
        suit_counts = {}
        for card in cards:
            suit_counts[card['suit']] = suit_counts.get(card['suit'], 0) + 1
        is_flush = max(suit_counts.values()) >= 5

        # Check for straight
        values = sorted([c['value'] for c in cards], reverse=True)
        is_straight = self._check_straight(values)

        # Count ranks
        rank_counts = {}
        for card in cards:
            rank_counts[card['rank']] = rank_counts.get(card['rank'], 0) + 1

        sorted_counts = sorted(rank_counts.values(), reverse=True)

        # Determine hand type
        if is_straight and is_flush:
            if values[0] == 14:  # Ace high
                hand_type = 'Royal Flush'
            else:
                hand_type = 'Straight Flush'
        elif sorted_counts[0] == 4:
            hand_type = 'Four of a Kind'
        elif sorted_counts[0] == 3 and sorted_counts[1] == 2:
            hand_type = 'Full House'
        elif is_flush:
            hand_type = 'Flush'
        elif is_straight:
            hand_type = 'Straight'
        elif sorted_counts[0] == 3:
            hand_type = 'Three of a Kind'
        elif sorted_counts[0] == 2 and sorted_counts[1] == 2:
            hand_type = 'Two Pair'
        elif sorted_counts[0] == 2:
            hand_type = 'One Pair'
        else:
            hand_type = 'High Card'

        return {
            'hand_type': hand_type,
            'hand_rank': self.HAND_RANKINGS[hand_type],
            'high_card': cards[0]['rank'],
            'method': 'basic'
        }

    def _check_straight(self, values):
        """Check if values form a straight."""
        # Remove duplicates and sort
        unique_values = sorted(set(values), reverse=True)

        # Check for 5 consecutive cards
        for i in range(len(unique_values) - 4):
            if unique_values[i] - unique_values[i + 4] == 4:
                return True

        # Check for wheel (A-2-3-4-5)
        if 14 in unique_values and 2 in unique_values and 3 in unique_values and 4 in unique_values and 5 in unique_values:
            return True

        return False

    def evaluate_hand(self, hole_cards, community_cards):
        """
        Evaluate hand strength.

        Args:
            hole_cards (list): List of 2 hole cards (e.g., ["A♠", "K♠"])
            community_cards (list): List of 3-5 community cards

        Returns:
            dict: Hand evaluation result
        """
        if self.use_treys:
            return self.evaluate_hand_treys(hole_cards, community_cards)
        else:
            return self.evaluate_hand_basic(hole_cards, community_cards)

    def evaluate_hand_treys(self, hole_cards, community_cards):
        """Evaluate hand using treys library."""
        # Convert to treys format
        def convert_card(card_str):
            rank, suit = self.parse_card(card_str)
            # Map suit symbols to treys format
            suit_map = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
            rank_map = {'10': 'T'}
            treys_rank = rank_map.get(rank, rank)
            treys_suit = suit_map.get(suit, suit.lower())
            return self.Card.new(treys_rank + treys_suit)

        board = [convert_card(c) for c in community_cards]
        hand = [convert_card(c) for c in hole_cards]

        # Evaluate
        score = self.treys_evaluator.evaluate(board, hand)
        hand_class = self.treys_evaluator.get_rank_class(score)
        hand_type = self.treys_evaluator.class_to_string(hand_class)

        return {
            'hand_type': hand_type,
            'hand_rank': 10 - hand_class,  # Invert to match our ranking
            'score': score,
            'method': 'treys'
        }


class PokerStrategy:
    """Implement poker decision-making strategy."""

    def __init__(self):
        """Initialize poker strategy."""
        self.evaluator = PokerHandEvaluator()

    def calculate_pot_odds(self, pot_size, bet_to_call):
        """
        Calculate pot odds.

        Args:
            pot_size (int): Current pot size
            bet_to_call (int): Amount to call

        Returns:
            float: Pot odds as a ratio
        """
        if bet_to_call == 0:
            return float('inf')
        return pot_size / bet_to_call

    def calculate_hand_strength(self, hole_cards, community_cards):
        """
        Calculate normalized hand strength (0-1).

        Args:
            hole_cards (list): Hole cards
            community_cards (list): Community cards

        Returns:
            float: Hand strength (0-1)
        """
        result = self.evaluator.evaluate_hand(hole_cards, community_cards)
        # Normalize hand rank to 0-1 scale
        return result['hand_rank'] / 10.0

    def recommend_action(self, hole_cards, community_cards, pot_size, bet_to_call,
                        stack_size, position='middle', num_players=6, strategy='balanced'):
        """
        Recommend poker action based on game state.

        Args:
            hole_cards (list): Your hole cards
            community_cards (list): Community cards on board
            pot_size (int): Current pot size
            bet_to_call (int): Amount needed to call
            stack_size (int): Your remaining chips
            position (str): Your position ('early', 'middle', 'late', 'button', 'sb', 'bb')
            num_players (int): Number of players at table
            strategy (str): Strategy style ('aggressive', 'balanced', 'conservative')

        Returns:
            dict: Recommended action and reasoning
        """
        # Evaluate hand
        hand_eval = self.evaluator.evaluate_hand(hole_cards, community_cards)
        hand_strength = hand_eval['hand_rank'] / 10.0

        # Calculate pot odds
        pot_odds = self.calculate_pot_odds(pot_size, bet_to_call)

        # Determine action based on strategy
        action = self._determine_action(
            hand_strength, pot_odds, bet_to_call, stack_size,
            position, strategy, len(community_cards)
        )

        return {
            'action': action['action'],
            'amount': action.get('amount', 0),
            'reasoning': action['reasoning'],
            'hand_evaluation': hand_eval,
            'hand_strength': hand_strength,
            'pot_odds': pot_odds
        }

    def _determine_action(self, hand_strength, pot_odds, bet_to_call, stack_size,
                         position, strategy, board_size):
        """Internal method to determine best action."""
        
        # Pre-flop (no community cards)
        if board_size == 0:
            return self._preflop_action(hand_strength, bet_to_call, position, strategy)

        # Post-flop action
        # Strong hands
        if hand_strength >= 0.8:
            if bet_to_call > stack_size * 0.3:
                return {'action': 'call', 'reasoning': 'Strong hand, pot committed'}
            else:
                bet_amount = min(stack_size, pot_odds * 3)
                return {
                    'action': 'raise',
                    'amount': bet_amount,
                    'reasoning': 'Strong hand, betting for value'
                }

        # Good hands
        elif hand_strength >= 0.6:
            if bet_to_call < stack_size * 0.1:
                return {'action': 'call', 'reasoning': 'Good hand, cheap to see next card'}
            elif pot_odds > 3:
                return {'action': 'call', 'reasoning': 'Good hand with good pot odds'}
            else:
                return {'action': 'fold', 'reasoning': 'Good hand but poor pot odds'}

        # Medium hands
        elif hand_strength >= 0.4:
            if bet_to_call == 0:
                return {'action': 'check', 'reasoning': 'Medium hand, see next card for free'}
            elif bet_to_call < stack_size * 0.05 and pot_odds > 5:
                return {'action': 'call', 'reasoning': 'Medium hand with excellent pot odds'}
            else:
                return {'action': 'fold', 'reasoning': 'Medium hand, not worth the cost'}

        # Weak hands
        else:
            if bet_to_call == 0:
                return {'action': 'check', 'reasoning': 'Weak hand, check to see next card'}
            else:
                return {'action': 'fold', 'reasoning': 'Weak hand, not worth calling'}

    def _preflop_action(self, hand_strength, bet_to_call, position, strategy):
        """Determine pre-flop action."""
        # This is simplified - real pre-flop strategy is more complex
        if hand_strength >= 0.8:
            return {
                'action': 'raise',
                'amount': bet_to_call * 3,
                'reasoning': 'Premium hand, raising for value'
            }
        elif hand_strength >= 0.6:
            if position in ['late', 'button']:
                return {
                    'action': 'raise' if strategy == 'aggressive' else 'call',
                    'reasoning': 'Good hand in good position'
                }
            else:
                return {'action': 'call', 'reasoning': 'Good hand, calling to see flop'}
        elif hand_strength >= 0.4:
            if bet_to_call < 3:  # Small bet
                return {'action': 'call', 'reasoning': 'Speculative hand, cheap to see flop'}
            else:
                return {'action': 'fold', 'reasoning': 'Marginal hand, too expensive'}
        else:
            return {'action': 'fold', 'reasoning': 'Weak hand, folding'}


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Poker decision logic and hand evaluation"
    )
    parser.add_argument(
        "--hand",
        required=True,
        help="Your hole cards (e.g., 'A♠ K♠' or 'A♠,K♠')"
    )
    parser.add_argument(
        "--board",
        default="",
        help="Community cards (e.g., 'Q♠ J♠ 10♠')"
    )
    parser.add_argument(
        "--pot",
        type=int,
        default=0,
        help="Current pot size"
    )
    parser.add_argument(
        "--bet",
        type=int,
        default=0,
        help="Bet amount to call"
    )
    parser.add_argument(
        "--stack",
        type=int,
        default=1000,
        help="Your stack size"
    )
    parser.add_argument(
        "--position",
        choices=['early', 'middle', 'late', 'button', 'sb', 'bb'],
        default='middle',
        help="Your table position"
    )
    parser.add_argument(
        "--strategy",
        choices=['aggressive', 'balanced', 'conservative'],
        default='balanced',
        help="Playing strategy"
    )

    args = parser.parse_args()

    # Parse cards
    hole_cards = [c.strip() for c in args.hand.replace(',', ' ').split()]
    community_cards = [c.strip() for c in args.board.replace(',', ' ').split()] if args.board else []

    # Initialize strategy
    strategy = PokerStrategy()

    # Evaluate hand
    print("\n" + "="*50)
    print("HAND EVALUATION")
    print("="*50)
    print(f"Hole Cards: {' '.join(hole_cards)}")
    if community_cards:
        print(f"Board: {' '.join(community_cards)}")
    
    hand_eval = strategy.evaluator.evaluate_hand(hole_cards, community_cards)
    print(f"\nHand Type: {hand_eval['hand_type']}")
    print(f"Hand Rank: {hand_eval['hand_rank']}/10")

    # Get recommendation
    if args.pot > 0 or args.bet > 0:
        print("\n" + "="*50)
        print("DECISION RECOMMENDATION")
        print("="*50)
        print(f"Pot Size: {args.pot}")
        print(f"Bet to Call: {args.bet}")
        print(f"Stack Size: {args.stack}")
        print(f"Position: {args.position}")
        print(f"Strategy: {args.strategy}")

        recommendation = strategy.recommend_action(
            hole_cards, community_cards,
            args.pot, args.bet, args.stack,
            args.position, strategy=args.strategy
        )

        print(f"\n→ Recommended Action: {recommendation['action'].upper()}")
        if recommendation['amount'] > 0:
            print(f"→ Amount: {recommendation['amount']}")
        print(f"→ Reasoning: {recommendation['reasoning']}")
        print(f"→ Pot Odds: {recommendation['pot_odds']:.2f}")

        # Save to JSON
        with open('decision.json', 'w') as f:
            json.dump(recommendation, f, indent=2)
        print("\n✓ Detailed recommendation saved to decision.json")


if __name__ == "__main__":
    main()
