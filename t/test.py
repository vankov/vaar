from encoding import VectorEncoder
from predicates import Atom, Property, Predicate
import tools

john = Atom("john")     
marry = Atom("marry")     
peter = Atom("peter")
maria = Atom("maria")
driver = Property("driver", [john])
loves1 = Predicate("loves", 2, [john], [marry])
loves2 = Predicate("loves", 2, [marry], [driver])
encoder = VectorEncoder(5, 2, 10)

vec_repr, predicates = encoder.encode_predicates([loves1, driver])
tools.print_v(vec_repr, predicates=predicates)

#tools.print_v(encoder.encode_predicates([loves2]))