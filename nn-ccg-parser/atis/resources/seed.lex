what :- S/N : (lambda $0:<e,t> $0)
what :- S/(S\NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
what :- S/(S/NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
what :- S/NP : (lambda $0:e $0)
which :- S/NP : (lambda $0:e $0)
which :- S/N : (lambda $0:<e,t> $0)
which :- S/(S\NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
which :- S/(S/NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
// where is lester pearson airport
where :- S/NP : (lambda $0:e (lambda $1:e (loc:<lo,<lo,t>> $0 $1)))
// where is mke located
where :- S/(S/PP) : (lambda $0:<e,t> $0)

list :- S/N : (lambda $0:<i,t> $0)

//null
a :- N/N : (lambda $0:<e,t> $0)
does :- (S\NP)/(S\NP) : (lambda $0:<e,t> $0)
does :- (S/NP)/(S/NP) : (lambda $0:<e,t> $0)
//information :- N\N : (lambda $0:<e,t> $0)
a :- NP/NP : (lambda $0:e $0)
please :- NP\NP : (lambda $0:e $0)
please :- S\S : (lambda $0:t $0)
please :- S/S : (lambda $0:t $0)
available :- PP/PP : (lambda $0:<e,t> $0)

//wh-words
show :- S/N : (lambda $0:<e,t> $0)
list :- S/N : (lambda $0:<e,t> $0)
give :- S/N : (lambda $0:<e,t> $0)  
// should be S/PP, but changed "to" to N/(S\NP)
i need :- S/N : (lambda $0:<e,t> $0)  

how many :- S/(S\NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (count:<<e,t>,i> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
how many :- S/(S/NP)/N : (lambda $0:<e,t> (lambda $1:<e,t> (count:<<e,t>,i> (lambda $2:e (and:<t*,t> ($0 $2) ($1 $2)))))
how many :- S/N : (lambda $0:<e,t> (count:<<e,t>,i> $0))
//how many :- S/NP : (lambda $0:e $0)
// how many people :- S/(S\NP) : (lambda $0:e (lambda $1:i (=:<i,<i,t>> (capapcity:<e,i> $0) $1)))

how much :- S/(S/NP) : (lambda $0:<e,t> $0)
// how much :- S/NP : (lambda $0:e (lambda $1:i (=:<i,<i,t>> (cost:<tr,i> $0) $1)))
// how expensive :- S/NP : (lambda $0:e (lambda $1:i (=:<i,<i,t>> (cost:<tr,i> $0) $1)))

// what time :- S/NP : (lambda $0:e (lambda $1:i (=:<i,<i,t>> (departure_time:<tr,i> $0) $1)))
// how much time :- S/NP : (lambda $0:e (lambda $1:i (=:<i,<i,t>> (minutes_distant:<e,i> $0) $1)))

//how :- S/NP : (lambda $0:e $0)

// Nouns
flights :- N : (lambda $0:e (flight:<fl,t> $0))
flight :- N : (lambda $0:e (flight:<fl,t> $0))
ground transportation :- N : (lambda $0:e (ground_transport:<gt,t> $0))
cities :- N : (lambda $0:e (city:<ci,t> $0))
meals :- N : (lambda $0:e (meal:<me,t> $0))
airlines :- N : (lambda $0:e (airline:<al,t> $0))
aircrafts :- N : (lambda $0:e (aircraft_code:<ac,t> $0))
car rental :- N : (lambda $0:e (rental_car:<gt,t> $0)) 
// ticket doesn't have meaning
ticket :- N : (lambda $0:e true:t) 

// preposition
from :- PP/NP : (lambda $0:e (lambda $1:e (from:<fl,<lo,t>> $1 $0)))  
to :- PP/NP : (lambda $0:e (lambda $1:e (to:<fl,<lo,t>> $1 $0)))
on :- PP/NP : (lambda $0:e (lambda $1:e (day:<tr,<da,t>> $1 $0))) 
// to boston and then to atlanta...
to :- PP/NP : (lambda $0:e (lambda $1:e (stop:<fl,<lo,t>> $1 $0)))
with a stop :- PP/NP : (lambda $0:e (lambda $1:e (stop:<fl,<lo,t>> $1 $0)))
less than :- PP/NP : (lambda $0:e (lambda $1:e (<:<i,<i,t>> (cost:<tr,i> $1) $0)))
round trip :- PP : (lambda $0:e (round_trip:<fl,t> $0))
on :- PP/NP : (lambda $0:e (lambda $1:e (airline2:<fl,<al,t>> $1 $0))) 
 //on :- PP/NP/NP : (lambda $0:e (lambda $1:e (lambda $2:e (and:<t*,t> (month:<tr,<mn,t>> $2 $0) (day_number:<tr,<dn,t>> $2 $1))))) 
at :- PP/NP : (lambda $0:e (lambda $1:e (=:<i,<i,t>> (departure_time:<tr,i> $1) $0)))
after :- PP/NP : (lambda $0:e (lambda $1:e (>:<i,<i,t>> (departure_time:<tr,i> $1) $0)))
before :- PP/NP : (lambda $0:e (lambda $1:e (<:<i,<i,t>> (departure_time:<tr,i> $1) $0)))
//between :- PP/NP/NP : (lambda $0:e (lambda $1:e (lambda $2:e (and:<t*,t> (from:<fl,<lo,t>> $2 $0) (to:<fl,<lo,t>> $2 $1)))))
//between :- PP/NP/NP : (lambda $0:e (lambda $1:e (lambda $2:e (and:<t*,t> (>:<i,<i,t>> (departure_time:<tr,i> $2) $0) (<:<i,<i,t>> (departure_time:<tr,i> $2) $1)))))
// in the morning
in :- PP/NP : (lambda $0:e (lambda $1:e (during_day:<tr,<pd,t>> $1 $0)))  
from the airport :- PP/NP : (lambda $0:e (lambda $1:e (from_airport:<gt,<lo,t>> $1 $0)))
from :- PP/NP : (lambda $0:e (lambda $1:e (from_airport:<gt,<lo,t>> $1 $0)))
to :- PP/NP : (lambda $0:e (lambda $1:e (to_city:<gt,<ci,t>> $1 $0)))  
by way of :- PP/NP : (lambda $0:e (lambda $1:e (stop:<fl,<lo,t>> $1 $0)))  


// adjective
one way :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (oneway:<tr,t> $1))))
nonstop :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (nonstop:<fl,t> $1))))  
round trip :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (round_trip:<fl,t> $1))))
nonstop :- PP : (lambda $1:e (nonstop:<fl,t> $1))
tomorrow :- PP : (lambda $0:e (tomorrow:<tr,t> $0))

// superlatives
earliest :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (departure_time:<tr,i> $1))))
first :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (departure_time:<tr,i> $1))))
latest :- NP/N : (lambda $0:<e,t> (argmax:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (departure_time:<tr,i> $1))))
last :- NP/N : (lambda $0:<e,t> (argmax:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (departure_time:<tr,i> $1))))
shortest :- NP/N : (lambda $0:<e,t> (argmax:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (time_elapsed:<e,i> $1))))
highest :- NP/N :  (lambda $0:<e,t> (argmax:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (cost:<tr,i> $1))))
cheapest :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (cost:<tr,i> $1))))
lowest :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (cost:<tr,i> $1))))
smallest :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (capacity:<e,i> $1))))
greatest :- NP/N : (lambda $0:<e,t> (argmax:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (capacity:<e,i> $1))))
most expensive :- NP/N : (lambda $0:<i,t> (max:<<i,t>,i> $0))
// what airport in new york is closest to downtown
//closest :- NP/N : (lambda $0:<e,t> (argmin:<<e,t>,<<e,i>,e>> (lambda $1:e ($0 $1)) (lambda $1:e (miles_distant:<e,i> $1))))

// fare from boston to new york
fare :- N/PP : (lambda $0:<e,t> (lambda $1:i (exists:<<e,t>,t> (lambda $2:e (and:<t*,t> ($0 $2) (=:<i,<i,t>> (fare:<fl,i> $2) $1))))))

// nonstop fare from boston to new york
fare :- N\(N/N)/PP : (lambda $0:<e,t> (lambda $3:<<e,t>,<e,t>> (lambda $1:i (exists:<<e,t>,t> (lambda $2:e (and:<t*,t> ($3 $0 $2) (=:<i,<i,t>> (fare:<fl,i> $2) $1)))))))

// DEG
expensive :- DEG : (lambda $0:e (cost:<tr,i> $0)) 
capacity :- DEG : (lambda $0:e (capacity:<e,i> $0)) 
//what is the lowest price fare from...
price :- DEG : (lambda $0:e (cost:<tr,i> $0)) 
cost :- DEG : (lambda $0:e (cost:<tr,i> $0)) 

//verbs
//how much does a first class flight cost from... airlines servicing between...
cost :- (S\NP)/PP : (lambda $0:i (lambda $1:e (=:<i,<i,t>> (cost:<tr,i> $1) $0)))
have :- (S\NP)/NP : (lambda $0:e (lambda $1:e (airline2:<fl,<al,t>> $0 $1))) 
//how many airports does oakland have
have :- (S\NP)/NP : (lambda $0:e (lambda $1:e (loc:<lo,<lo,t>> $0 $1))) 
serves :- (S\NP)/NP : (lambda $0:e (lambda $1:e (services:<al,<lo,t>> $1 $0))) 

total :- NP/N/DEG : (lambda $0:<e,i> (lambda $1:<e,t> (sum:<<e,t>,<<e,i>,i>> (lambda $2:e ($1 $2)) (lambda $2:e ($0 $2)))))
total :- NP/N : (lambda $0:<e,t> (count:<<e,t>,i> $0))

// that 
that :- PP/(S\NP) : (lambda $0:<e,t> $0)
that :- PP/(S/NP) : (lambda $0:<e,t> $0)
which :- PP/(S\NP) : (lambda $0:<e,t> $0)
which :- PP/(S/NP) : (lambda $0:<e,t> $0)




// most expensive...
most :- NP/N/DEG : (lambda $0:<e,i> (lambda $1:<e,t> (argmax:<<e,t>,<<e,i>,e>> $1 $0)))
least :- NP/N/DEG : (lambda $0:<e,i> (lambda $1:<e,t> (argmin:<<e,t>,<<e,i>,e>> $1 $0)))

// for non-maximal factoring
houston :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (loc:<lo,<lo,t>> $1 houston:ci))))
houston :- PP : (lambda $0:e (loc:<lo,<lo,t>> $0 houston:ci))

dinner :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (meal2:<fl,<me,t>> $1 dinner:me))))
dinner :- PP : (lambda $0:e (meal2:<fl,<me,t>> $0 dinner:me))
//dinner :- N : (lambda $0:e (meal2:<fl,<me,t>> $0 dinner:me))

747 :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (aircraft_code2:<fl,<ac,t>> $1 747:ac))))
747 :- PP : (lambda $0:e (aircraft_code2:<fl,<ac,t>> $0 747:ac))
747 :- N : (lambda $0:e (aircraft_code2:<fl,<ac,t>> $0 747:ac))

boeing :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (manufacturer:<e,<mf,t>> $1 boeing:mf)))) 
boeing :- PP : (lambda $0:e (manufacturer:<e,<mf,t>> $0 boeing:mf))
boeing :- N : (lambda $0:e (manufacturer:<e,<mf,t>> $0 boeing:mf))

morning :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (during_day:<tr,<pd,t>> $1 morning:pd))))
morning :- PP : (lambda $0:e (during_day:<tr,<pd,t>> $0 morning:pd))
morning :- N : (lambda $0:e (during_day:<tr,<pd,t>> $0 morning:pd))

monday :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (day:<tr,<da,t>> $1 monday:da))))
monday :- PP : (lambda $0:e (day:<tr,<da,t>> $0 monday:da))
monday :- N : (lambda $0:e (day:<tr,<da,t>> $0 monday:da))

august :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (month:<tr,<mn,t>> $1 august:mn))))
august :- PP : (lambda $0:e (month:<tr,<mn,t>> $0 august:mn))
august :- N : (lambda $0:e (month:<tr,<mn,t>> $0 august:mn))

united :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (airline2:<fl,<al,t>> $1 ua:al))))
united :- PP : (lambda $0:e (airline2:<fl,<al,t>> $0 ua:al))
united :- N : (lambda $0:e (airline2:<fl,<al,t>> $0 ua:al))

746 :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (flight_number:<fl,<fn,t>> $1 746:fn))))
746 :- PP : (lambda $0:e (flight_number:<fl,<fn,t>> $0 746:fn))

7 :- N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (day_number:<tr,<dn,t>> $1 7:dn))))
7 :- PP : (lambda $0:e (day_number:<tr,<dn,t>> $0 7:dn))

705 :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (=:<i,<i,t>> (departure_time:<tr,i> $1) 705:ti))))
705 :- PP : (lambda $0:e (=:<i,<i,t>> (departure_time:<tr,i> $0) 705:ti))

705 :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (=:<i,<i,t>> (arrival_time:<tr,i> $1) 705:ti))))
705 :- PP : (lambda $0:e (=:<i,<i,t>> (arrival_time:<tr,i> $0) 705:ti))

boston :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (from:<fl,<lo,t>> $1 boston:ci))))
boston :- PP : (lambda $0:e (from:<fl,<lo,t>> $0 boston:ci))

boston :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (to:<fl,<lo,t>> $1 boston:ci))))
boston :- PP : (lambda $0:e (to:<fl,<lo,t>> $0 boston:ci))

boston :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (stop:<fl,<lo,t>> $1 boston:ci))))
boston :- PP : (lambda $0:e (stop:<fl,<lo,t>> $0 boston:ci))

boston :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (from_airport:<gt,<lo,t>> $1 boston:ci))))
boston :- PP : (lambda $0:e (from_airport:<gt,<lo,t>> $0 boston:ci))

boston :-  N/N : (lambda $0:<e,t> (lambda $1:e (and:<t*,t> ($0 $1) (to_city:<gt,<ci,t>> $1 boston:ci))))
boston :- PP : (lambda $0:e (to_city:<gt,<ci,t>> $0 boston:ci))
