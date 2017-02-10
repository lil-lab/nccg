///////////////////////////from np-fixed.nav
// numbers
1 :- NP : 1:n
2 :- NP : 2:n
3 :- NP : 3:n
4 :- NP : 4:n
5 :- NP : 5:n
90 :- NP : 90:n
one :- NP : 1:n
two :- NP : 2:n
three :- NP : 3:n
four :- NP : 4:n
five :- NP : 5:n
ninety :- NP : 90:n
first :- NP : 1:n
second :- NP : 2:n
twice :- NP : 2:n
third :- NP : 3:n
fourth :- NP : 4:n
fifth :- NP : 5:n

// ordinals.  Use e rather than loc for generality.  "distance"is domain-special.
//first :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 1:n)
//one :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 1:n)
//next :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 1:n)
//second :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 2:n)
//third :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 3:n)
//fourth :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 4:n)
//fifth :- NP/(S|NP) : (lambda $0:<e,t> (n-th:e $0 (lambda $1:e (distance:n current-loc:loc $1)) 5:n)

// relative locations. But, we're phasing out "-loc".
/////right
//on the right :- NP : right-loc:loc
//on right :- NP : right-loc:loc
//right :- NP : right-loc:loc
//to the right :- NP : right-loc:loc
/////left
//on the left :- NP : left-loc:loc
//on left :- NP : left-loc:loc
//left :- NP : left-loc:loc
//to the left :- NP : left-loc:loc
/////forward
//in front of you :- NP : forward-loc:loc
//forward :- NP : forward-loc:loc
//move forward :- NP : forward-loc:loc
//straight :- NP : forward-loc:loc

// S|NP is syntactic type of a set (S|NP === N).
// These are the sets of things that are left/etc of [..?]
//left :- S|NP : (lambda $0:e (!=:t (left-of:loc $0) null:e))
//right :- S|NP : (lambda $0:e (!=:t (right-of:loc $0) null:e))
//ahead :- S|NP : (lambda $0:e (!=:t (forward-of:loc $0) null:e))
//forward :- S|NP : (lambda $0:e (!=:t (forward-of:loc $0) null:e))
//straight :- S|NP : (lambda $0:e (!=:t (forward-of:loc $0) null:e))

// For every unary typing predicate taking a 'e' and returning truth value, create
// sets of all intersections, rooms, doors, etc.  These rely on the functions
// junction:t, etc being defined in nav.types.
//room :- S|NP : (lambda $0:e (room:t $0))
//hall :- S|NP : (lambda $0:e (hall:t $0))
//hallway :- S|NP : (lambda $0:e (hall:t $0))
//corridor :- S|NP : (lambda $0:e (hall:t $0))
//junction :- S|NP : (lambda $0:e (junction:t $0))
//intersection :- S|NP : (lambda $0:e (junction:t $0))
//doorway :- S|NP : (lambda $0:e (junction:t $0))
//door :- S|NP : (lambda $0:e (junction:t $0))

// verbs are the core of a clause in the imperative
//go :- S/NP : (lambda x:e (move x))
//move :- S/NP : (lambda x:e (move x))
//go :- S/NP : (lambda $0:loc (move-to:t $0))
//move :- S/NP : (lambda $0:loc (move-to:t $0))
//enter :- S/NP : (lambda $0:loc (lambda $1:loc (and:t (room:t $1) (move-to:t $0))))
//leave :- S/NP : (lambda $0:loc (lambda $1:loc (and:t (room:t $1) (move-to:t $0))))
//take :- S/NP : (lambda $0:loc (move-to:t $0))

// prepositions
//into :- NP/NP :  (lambda $0:loc (lambda $1:loc (and:t (room:t $1) (move-to:t $0))))
//to :- NP/NP : (lambda $0:e (move-to:t $0))
//to :- NP/NP : (lambda $0:e $0)

///////////////////////////from english.lex

// Basic lexical items
//a :- NP/NP : (lambda $0:<e,t> $0)
a :- NP/NP : (lambda $0:e $0)
the :- NP/NP : (lambda $0:e $0)
your :- NP/NP : (lambda $0:e $0)

// No-ops/semantic sugar
//of the hall :- S\S : (lambda $0:t $0)

// Coordinating different objects in adjacent sentences
//and :- S\S/S : (lambda $0:t (lambda $1:t (do-sequentially:t $1 $0)))

