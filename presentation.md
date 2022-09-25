---
title: Breadth-First Traversals via Staging
patat:
  theme:
    codeBlock: [vividBlack]
    code: [vividBlack]
  incrementalLists: true

...

# Introduction

## Overview

- Jeremy Gibbons, *Oisín Kidney*, Tom Schrijvers, and Nicolas Wu

- **Traversals** of **Trees**

- Topics

    * Basic traversals, algorithms

    * Problems like *repmin*, *sort tree*
    
    * *Applicatives*, *Traversables*, and *Free Applicatives*

- Takeaway: a technique for **staging** effectful computations

    * The ability to **reorder** effects and values, **independently** of 
      each other.



## Important Types

. . .

```haskell
data Tree   a = a :& [Tree a]

type Forest a = [Tree a]
```

. . .

```haskell
  3 :& [ 1 :& [ 1 :& []            --      3─┬─1─┬─1
              , 5 :& []]           --        │   └─5
       , 4 :& [ 9 :& []            --        └─4─┬─9
              , 2 :& []]]          --            └─2
```

## Traversal Orders

```
   breadth-first                depth-first


   3──┬──1──┬──1               3──┬──1──┬──1
      │     │                     │     │
      │     │                     │     │
      │     └──5                  │     └──5
      │                           │
      │                           │
      └──4──┬──9                  └──4──┬──9
            │                           │
            │                           │
            └──2                        └──2
```


## Traversal Orders

```
   breadth-first                depth-first
   ↓     ↓     ↓
 ┏━━━┓ ┏━━━┓ ┏━━━┓           ┏━━━━━━━━━━━━━━━┓
 ┃ 3─╂┬╂─1─╂┬╂─1 ┃         → ┃ 3──┬──1──┬──1 ┃
 ┗━━━┛│┃   ┃│┃   ┃           ┗━━━━┿━━━━━┿━━━━┛
      │┃   ┃│┃   ┃                │     │┏━━━┓
      │┃   ┃└╂─5 ┃         →      │     └╂─5 ┃
      │┃   ┃ ┃   ┃                │      ┗━━━┛
      │┃   ┃ ┃   ┃                │┏━━━━━━━━━┓
      └╂─4─╂┬╂─9 ┃         →      └╂─4──┬──9 ┃
       ┗━━━┛│┃   ┃                 ┗━━━━┿━━━━┛
            │┃   ┃                      │┏━━━┓
            └╂─2 ┃         →            └╂─2 ┃
             ┗━━━┛                       ┗━━━┛
```

<!-- 

. . .

```haskell
dfe :: Tree a -> [a]
dfe (x :& xs) = x : concatMap dfe xs
```

. . .

```haskell
    ╭                         ╮
dfe │ 3 :& [ 1 :& [ 1 :& []   │ = [3,1,1,5,4,9,2]
    │             , 5 :& []]  │
    │      , 4 :& [ 9 :& []   │
    │             , 2 :& []]] │
    ╰                         ╯
``` 


-->

## Applicative and Traversable

. . .

```haskell
class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
  
```

. . .

```haskell
(⊗)  :: Applicative f => f a  -> f b  -> f (a, b)
(<*) :: Applicative f => f a  -> f () -> f a
(*>) :: Applicative f => f () -> f a  -> f a
```

. . .

```haskell
class Foldable t => Traversable t where
  traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
```


## Renumbering with Traverse

. . .

```haskell
renumber :: Tree a -> Tree Int
```

. . .

```haskell
get       ::                 State Int Int
modify    :: (Int -> Int) -> State Int ()
evalState :: State Int a -> Int -> a
```

. . .

```haskell
instance Traversable Tree where ...
```

. . .

```haskell
renumber t = evalState (traverse num t) 1
  where num _ = get <* modify succ
```

## Renumbering with Traverse

```haskell
         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 3 :& []
         │             , 5 :& []]  │               , 4 :& []]
         │      , 4 :& [ 9 :& []   │        , 5 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
```


<!-- 
## Bird's repmin problem

. . .

```haskell
       ╭                         ╮
repmin │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 1 :& [ 1 :& []
       │             , 5 :& []]  │               , 1 :& []]
       │      , 4 :& [ 9 :& []   │        , 1 :& [ 1 :& []
       │             , 2 :& []]] │               , 1 :& []]]
       ╰                         ╯
```

. . .

```haskell
repmin :: Tree Int -> Tree Int
repmin t = let m = minimum t in fmap (const m) t
```

## Repmin as a Traverse


- 2 operations: minimum, and replacing.

- Min can be done with the writer effect, and the "min" monoid.

- Replace can be done with the reader effect.

## Replace

. . .

```haskell
ask       :: Reader e e
runReader :: Reader e a -> e -> a
```

. . .

```haskell
replace :: Traversable t => t a -> e -> t e
replace t = runReader (traverse (const ask) t)
```


## Minimum

. . .

```haskell
tell       :: Monoid w => w -> Writer w ()
execWriter :: Writer w a -> w

execWriter (x <*> y) = execWriter x <> execWriter y
```

. . .

```haskell
data ⌈ a ⌉ = In a | Top
instance Ord a => Monoid ⌈ a ⌉ where (<>) = min
inBound (In x) = x
```

. . .

```haskell
minimum :: (Ord a, Traversable t) => t a -> a
minimum = inBound . execWriter . traverse (tell . In)
```

## Combining

```haskell
repmin :: (Ord a, Traversable t) => t a -> t a
repmin t = replace t (minimum t)
```

## Day Convolution


```haskell
data Day f g a where
  (:<*>) :: f (a -> b) -> g a -> Day f g b

instance (Applicative f, Applicative g) => Applicative (Day f g) where ...

phase1 :: Applicative g => f a -> Day f g a
phase2 :: Applicative f => g a -> Day f g a
```

## Day Convolution for Fusing

```haskell
repmin  :: (Traversable t, Ord a) => t a -> t a


repmin  t = replace t (minimum t)
```

## Day Convolution for Fusing

```haskell
repmin  :: (Traversable t, Ord a) => t a -> t a


repmin  t = 
  runReader  (traverse (\_ -> inBound <$> ask) t) $
  execWriter (traverse (\x -> tell (In x)) t)
```

## Day Convolution for Fusing

```haskell
repminE :: (Traversable t, Ord a) 
        => t a 
        -> Day (Writer ⌈ a ⌉) (Reader ⌈ a ⌉) (t a)
repminE t = 
  phase2     (traverse (\_ -> inBound <$> ask) t) <*
  phase1     (traverse (\x -> tell (In x)) t)
```

. . .

```haskell
repmin :: (Traversable t, Ord a) => t a -> t a
repmin = loopDay . repminE

loopDay :: Day (Writer a) (Reader a) b -> b
loopDay (xs :<*> ys) = 
  let (x, e) = runWriter xs 
  in  x (runReader ys e)
```

## Day Convolution for Fusing

```haskell
repminE :: (Traversable t, Ord a) 
        => t a 
        -> Day (Writer ⌈ a ⌉) (Reader ⌈ a ⌉) (t a)
repminE t = 
             (traverse (\_ -> phase2 (inBound <$> ask)) t) <*
             (traverse (\x -> phase1 (tell (In x))) t)
```

```haskell
repmin :: (Traversable t, Ord a) => t a -> t a
repmin = loopDay . repminE

loopDay :: Day (Writer a) (Reader a) b -> b
loopDay (xs :<*> ys) = 
  let (x, e) = runWriter xs 
  in  x (runReader ys e)
```

## Day Convolution for Fusing

```haskell
repminE :: (Traversable t, Ord a) 
        => t a 
        -> Day (Writer ⌈ a ⌉) (Reader ⌈ a ⌉) (t a)
repminE t = 
              traverse (\x -> phase2 (inBound <$> ask) <* 
                              phase1 (tell (In x))) t
```

```haskell
repmin :: (Traversable t, Ord a) => t a -> t a
repmin = loopDay . repminE

loopDay :: Day (Writer a) (Reader a) b -> b
loopDay (xs :<*> ys) = 
  let (x, e) = runWriter xs 
  in  x (runReader ys e)
```

# Multiple Phases 

-->

# Fusing traversals with Staging

## Sorting Tree Labels

```haskell
sortTree :: Ord a => Tree a -> Tree a
```

. . .

```haskell
         ╭                         ╮
sortTree │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 1 :& [ 2 :& []
         │             , 5 :& []]  │               , 3 :& []]
         │      , 4 :& [ 9 :& []   │        , 4 :& [ 5 :& []
         │             , 2 :& []]] │               , 9 :& []]]
         ╰                         ╯
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

. . .

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t)
```
. . .

```haskell
tree =  3 :& [  1 :& [  1 :& []         stack = []
                     ,  5 :& []]
             ,  4 :& [  9 :& []
                     ,  2 :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *> -- <--
                                 modify sort             *>
                                 traverse (\_ -> pop) t)
```

```haskell
tree =  3 :& [  1 :& [  1 :& []         stack = []
                     ,  5 :& []]
             ,  4 :& [  9 :& []
                     ,  2 :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *> -- <--
                                 modify sort             *>
                                 traverse (\_ -> pop) t)
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [3,1,1,5,9,4,2]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *> -- <--
                                 traverse (\_ -> pop) t)
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [3,1,1,5,9,4,2]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *> -- <--
                                 traverse (\_ -> pop) t)
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [1,1,2,3,4,5,9]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t)    -- <--
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [1,1,2,3,4,5,9]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t)    -- <--
```

```haskell
tree =  1 :& [  1 :& [  2 :& []         stack = []
                     ,  3 :& []]
             ,  4 :& [  5 :& []
                     ,  9 :& []]]
```

## Phases Type

. . .

```haskell
data Phases f a where
  Pure :: a -> Phases f a
  Link :: (x -> y -> a) -> f x -> Phases f y -> Phases f a
```

. . .


```haskell
instance Applicative f => Applicative (Phases f) where ...
```

## Phases Type: usage

. . .

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

. . .

```haskell


runPhases $            do phase 4 (putStrLn "a")
                          phase 2 (putStrLn "b")
                          phase 3 (putStrLn "c")
                          phase 1 (putStrLn "d")
                          phase 2 (putStrLn "e")
```

## Phases Type: usage

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

```haskell


runPhases $            do phase 4 (putStrLn "a")    --     > d
                          phase 2 (putStrLn "b")    --     > b
                          phase 3 (putStrLn "c")    --     > e
                          phase 1 (putStrLn "d")    --     > c
                          phase 2 (putStrLn "e")    --     > a
```

## Phases Type: usage

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

```haskell
out s = putStrLn s *> pure s

runPhases $ sequenceA $ [ phase 4 (out      "a")    --     > d
                        , phase 2 (out      "b")    --     > b
                        , phase 3 (out      "c")    --     > e
                        , phase 1 (out      "d")    --     > c
                        , phase 2 (out      "e") ]  --     > a
```

. . .

```haskell
["a","b","c","d","e"]
```

## Phases Type: Commutativity

```haskell
                         n /= m
-------------------------------------------------------------
  phase n x ⊗ phase m y = twist <$> (phase m y ⊗ phase n x)
```

```haskell
twist :: (a, b) -> (b, a)
```


## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t =
  flip evalState [] $
             traverse push t                      *>
             modify sort                          *>
             traverse (\_ -> pop) t)
 ```
 
## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 1 (traverse push t)                     *>
     phase 2 (modify sort)                         *>
     phase 3 (traverse (\_ -> pop) t)))
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
     phase 1 (traverse push t)                     *>
     phase 3 (traverse (\_ -> pop) t)))
```

. . .

```haskell
traverse (φ . f) = φ . traverse f
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
             (traverse (\x -> phase 1 (push x)) t) *>
             (traverse (\_ -> phase 3 pop) t)
```

## Commutativity

```haskell
                    f x ⊗ g y = twist <$> g y ⊗ f x
-------------------------------------------------------------------------
  traverse f t ⊗ traverse g t = unzip <$> traverse (\x -> f x ⊗ g x) t
```

```haskell
twist :: (a, b) -> (b, a)
unzip :: f (a, b) -> (f a, f b)
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
              traverse (\x -> phase 1 (push x)     *> 
                              phase 3 pop) t
```


<!--


```haskell
repmin t = let (u, m) = aux t m in u
  where
    aux :: Tree Int -> a -> (Tree a, Int)
    aux (x :& xs) m = (m :& ys, minimum (x : ms))
      where
        (ys, ms) = unzip (map aux xs)
```

. . .

```haskell
repmin t = let (u, m) = aux t in u m
  where
    aux :: Tree Int -> (a -> Tree a, Int)
    aux (x :& xs) = (\m -> m :& ys m, minimum (x : ms))
      where
        (ys, ms) = unzip (map aux xs)
```

---

```haskell
instance Monoid m => Applicative (m ,) where
  pure x = (mempty, x)
  (fm, f) <*> (xm, x) = (fm <> xm, f x)

data BoundedAbove a = In a | Top

instance Ord a => Monoid (BoundedAbove a) where
  mempty = Top
  Top <> x = x
  x <> Top = x
  In x <> In y = In (min x y)
  
getBounded :: BoundedAbove a -> a
getBounded (In x) = x

minimum :: Ord a => Tree a -> a
minimum = getBounded . fst . traverse (\x -> (In x, ()))
```

---

```haskell
instance Applicative (a ->) where
  pure x e = x
  (f <*> x) e = f e (x e)
  
replace :: Tree a -> b -> Tree b
replace = traverse (\_ e -> e)
```

---

```haskell
repmin t = replace t (minimum t)
```

--- 

---


---


```haskell
repminT :: (Traversable t, Ord a) => t a -> Day ((,) (BoundedAbove a)) ((->) (BoundedAbove a)) (t a)
repminT = traverse (\x -> phase1 (In x, ()) *> phase2 inBound)

runEnv :: Day ((,) e) ((->) e) a -> a
runEnv (Day c (e,xs) ys) = c xs (ys e)

repmin = runEnv . repminT
```

---

# Multiple Phases

```haskell
data Phases f a where
  Pure :: a -> Phases f a
  Link :: (x -> y -> a) -> f x -> Phases f y -> Phases f a
```

Notice that this is the free applicative.

But the applicative instance we're going to give it isn't:

```haskell
instance Applicative f => Applicative (Phases f) where
  pure = Pure
  Pure f <*> xs = fmap f xs
  fs <*> Pure x = fmap ($x) fs
  Link f x xs <*> Link g y ys = Link (\(l,r) (ls,rs) -> f l ls (g r rs)) (x ⊗ y) (xs ⊗ ys)
```


---


---
-->

# Breadth-First Traversals

## Breadth-First Enumeration

```haskell
levels :: Tree a → [[a]]
levels (x :& xs) = [x] : foldr (lzw (++)) [] (map levels xs)

lzw :: (a -> a -> a) -> [a] -> [a] -> [a]
lzw f (x:xs) (y:ys) = f x y : lzw f xs ys
lzw _ []     ys     = ys
lzw _ xs     []     = xs
```

. . .

```haskell
       ╭                         ╮
levels │ 3 :& [ 1 :& [ 1 :& []   │ = [[1],[1,4],[1,5,9,2]]
       │             , 5 :& []]  │
       │      , 4 :& [ 9 :& []   │
       │             , 2 :& []]] │
       ╰                         ╯
```


## Breadth-First Enumeration

```

   3──┬──1──┬──1
      │     │
      │     └──5
      │
      └──4──┬──9
            │
            └──2
```


## Breadth-First Enumeration

```
 ┏━━━┓ ┏━━━┓ ┏━━━┓
 ┃ 3─╂┬╂─1─╂┬╂─1 ┃
 ┗━━━┛│┃   ┃│┃   ┃
      │┃   ┃└╂─5 ┃
      │┃   ┃ ┃   ┃
      └╂─4─╂┬╂─9 ┃
       ┗━━━┛│┃   ┃
            └╂─2 ┃
             ┗━━━┛
```

## Breadth-First Traversal!

```haskell
bft :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
bft f = runPhases . go where 
  go (x :& xs) = (:&) <$> now (f x) <*> later (traverse go xs)
```

. . .

```haskell
renumber t = evalState (bft num t) 1 where num _ = get <* modify succ
```

. . .

```haskell
         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 4 :& []
         │             , 5 :& []]  │               , 5 :& []]
         │      , 4 :& [ 9 :& []   │        , 3 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
```

# Questions?
