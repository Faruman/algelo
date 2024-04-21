class EloSystem:
    """
    A class that represents an implementatin of the Elo Rating System with multiplicative margins of victory as proposed by Kovalchik 2020
    Adapted from Kraktoos's Elo Rating System (https://github.com/Kraktoos/Python-Elo-System)
    Written by Fabian Karst (fabian.karst@unisg.ch)
    """

    def __init__(self, base_elo: int = 1000, k: int = 32, use_mov:bool= False, mov_delta:int = 0, mov_alpha: float = 0):
        """
        Runs at Ini
        """
        self.base_elo = base_elo
        self.k = k
        self.use_mov = use_mov
        self.mov_delta = mov_delta
        self.mov_alpha = mov_alpha
        self.players = []

    # Player Methods

    def add_player(self, player: str, elo: int = None):
        """
        Adds the Player to the Players List, as well as their Elo
        Paramaters: Player, Elo (optional)
        Returns: None
        """
        info = {}
        if elo == None:
            elo = self.base_elo
        info["player"] = player
        info["elo"] = elo
        info["prob"] = 0
        self.players.append(info)
        self._update_everything()

    def remove_player(self, player: str):
        """
        Removes the Mentioned Player from the Players List
        Paramaters: Player
        Returns: None
        """
        for i in self.players:
            if i["player"] == player:
                self.players.remove(i)

    # Elo Methods
    def set_elo(self, player: str, elo: int):
        """
        Sets a Players Elo
        Paramaters: Player, Elo
        Returns: None
        """
        for i in self.players:
            if i["player"] == player:
                i["elo"] = int(elo)
        self._update_everything()

    def reset_elo(self, player: str):
        """
        Resets a Players Elo to the Base Elo
        Paramaters: Player
        Returns: None
        """
        for i in self.players:
            if i["player"] == player:
                i["elo"] = self.base_elo
        self._update_everything()

    def add_elo(self, player: str, elo: int):
        """
        Adds Elo to a Player
        Paramaters: Player, Elo
        Returns: None
        """
        for i in self.players:
            if i["player"] == player:
                i["elo"] += int(elo)
        self._update_everything()

    def remove_elo(self, player: str, elo: int):
        """
        Removes Elo of a Player
        Paramaters: Player, Elo
        Returns: None
        """
        for i in self.players:
            if i["player"] == player:
                i["elo"] -= int(elo)
        self._update_everything()

    # Return Methods
    def get_player_elo(self, player: str):
        """
        Returns the Player Elo
        Paramaters: Player
        Returns: Player Elo
        """
        for i in self.players:
            if i["player"] == player:
                return i["elo"]

    def get_player_prob(self, player: str):
        """
        Returns the Player Rank
        Paramaters: Player
        Returns: Player Rank
        """
        for i in self.players:
            if i["player"] == player:
                return i["prob"]

    def get_player_count(self):
        """
        Returns the Player Count
        Paramaters: None
        Returns: None
        """
        return len(self.players)

    # Return List Methods

    def get_overall_list(self):
        """
        Returns the Player, Elo and Ranks List
        Paramaters: None
        Returns: List of Dictionaries
        """
        elo_list = sorted(self.players, key=lambda d: d["elo"], reverse=True)
        return elo_list

    def get_players_with_elo(self, elo: int):
        """
        Returns a List of Players with the given Elo
        Paramaters: Elo
        Returns: List
        """
        players = []
        for i in self.players:
            if i["elo"] == elo:
                players.append(i["player"])
        return players

    # Main Matching System
    def record_match(self, player_a: str, player_b: str, winner: str = None, mov: int = 0):
        """
        Runs the Calculations and Updates the Score of both Player A and B Following a Simple Elo System
        Paramaters: Player A, Player B, Winner (if Winner is None it will be considered a draw)
        Returns: None
        """
        for i in self.players:
            if i["player"] == player_a:
                index_a = self.players.index(i)
                elo_a = i["elo"]
            elif i["player"] == player_b:
                index_b = self.players.index(i)
                elo_b = i["elo"]

        if winner == player_a:
            score_a = 1
            score_b = 0
        elif winner == player_b:
            score_a = 0
            score_b = 1
        else:
            score_a = 0.5
            score_b = 0.5

        # calculate the expected score
        pab = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))
        pba = 1 / (1 + 10 ** (-(elo_b - elo_a) / 400))

        # calculate the new elo
        if not self.use_mov:
            elo_a_new = elo_a + self.k * (score_a - pab)
            elo_b_new = elo_b + self.k * (score_b - pba)
        else:
            elo_a_new = elo_a + self.k * (1 + abs(mov / self.mov_delta)) ** self.mov_alpha * (score_a - pab)
            elo_b_new = elo_b + self.k * (1 + abs(mov / self.mov_delta)) ** self.mov_alpha * (score_b - pba)

        self.players[index_a]["elo"] = elo_a_new
        self.players[index_b]["elo"] = elo_b_new

        self._update_everything()

    def _update_everything(self):
        """
        Updates All Probs and Guarantees that Players don't get Negative Elo
        Paramaters: None
        Returns: None
        """
        for i in self.players:
            i["elo"] = int(i["elo"])
            if i["elo"] < 0:
                i["elo"] = 0
        for i in self.players:
            i["prob"] = sum([1 / (1 + 10 ** (-(i["elo"] - j["elo"]) / 400)) for j in self.players]) / len(self.players)