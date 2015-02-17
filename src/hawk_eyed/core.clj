(ns hawk-eyed.core
  (:gen-class))

(use '(incanter core stats))

;pass
(defn sigmoid [z]
  (/ 1 (+ 1 (Math/exp (- z)))))

;pass
(defn sigmoid-vec [m]
  (matrix (matrix-map #(sigmoid %) m)))

;pass
(defn sigmoid-prime [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

;pass
(defn sigmoid-prime-vec [m]
  (matrix (matrix-map #(sigmoid-prime %) m)))

(defrecord Network [num-layers
                    sizes
                    biases
                    weights])
;temp
(def net (Network. 3 
                   [2 3 1] 
                   (atom [(matrix [[0.1] [0.2] [0.3]]) (matrix [[0.4]])])
                   (atom [(matrix [[0.1 0.15] [0.2 0.25] [0.3 0.35]]) (matrix [[0.4 0.45 0.475]])])))
(def training-data [[(matrix [[1] [2]]) 2]
                    [(matrix [[2] [3]]) 3]
                    [(matrix [[3] [4]]) 4]
                    [(matrix [[4] [5]]) 5]])


(defn init-network [sizes]
  (let [num-layers (count sizes)
        biases (vec (for [y (rest sizes)] 
                      (if (= y 1)
                        (matrix [[(sample-normal y :mean 0 :sd 1)]])
                        (matrix (vec (for [bias (sample-normal y :mean 0 :sd 1)] [bias]))))))
        weights (loop [from (butlast sizes)
                       to (rest sizes)
                       c (- (count sizes) 1)
                       result []]
                  (if (zero? c)
                    result
                    (recur (rest from)
                           (rest to)
                           (dec c)
                           (conj result (matrix (vec (sample-normal (* (first from) (first to)))) (first from))))))]
      (Network. num-layers sizes (atom biases) (atom weights))))
                       

(defn feedforward [net a]
  (loop [b (deref (:biases net))
         w (deref (:weights net))
         c (- (:num-layers net) 1)
         result a]
    (if (zero? c)
      result
      (recur (rest b)
             (rest w)
             (dec c)
             (sigmoid-vec (plus (mmult (first w) result) (first b)))))))


(defn mini-batches [training-data mini-batch-size]
  (for [k (range 0 (count training-data) mini-batch-size)]
    (take mini-batch-size (drop k training-data))))


;pass
(defn cost-derivative [output-activations y]
  (minus output-activations y))

(defn backprop-f [net x]
  (loop [activation x
         activations [x]
         zs []
         b (deref (:biases net))
         w (deref (:weights net))]
    (if-not (first b)
      [zs activations]
      (let [z (plus (mmult (first w) activation) (first b))]
        (recur (sigmoid-vec z)
               (conj activations (sigmoid-vec z))
               (conj zs z)
               (rest b)
               (rest w))))))
(defn backprop-b [net zs activations delta nabla-b nabla-w]
  (loop [weights (deref (:weights net))
         zs (vec (butlast zs))
         as (vec (butlast (butlast activations)))
         nabla-b (conj (vec (butlast nabla-b)) delta)
         nabla-w (conj (vec (butlast nabla-w))
                       (mmult delta (trans (last (butlast activations)))))
         c 2]
    (if (= c (:num-layers net))
      [nabla-b nabla-w]
      (let [d (mult (mmult (trans (last weights)) delta) (sigmoid-prime-vec (last zs)))
            c-b (count nabla-b)
            c-w (count nabla-w)]
        (recur (vec (butlast weights))
               (vec (butlast zs))
               (vec (butlast as))
               (into (conj (vec (take (- c-b c) nabla-b)) d) 
                     (vec (drop (+ (- c-b c) 1) nabla-b)))
               (into (conj (vec (take (- c-w c) nabla-w)) 
                           (mmult d (trans (last as))))
                     (vec (drop (+ (- c-w c) 1) nabla-w)))
               (inc c))))))
(defn backprop [net x y]
  (let [nabla-b (for [b (deref (:biases net))] (matrix 0 (nrow b) (ncol b)))
        nabla-w (for [w (deref (:weights net))] (matrix 0 (nrow w) (ncol w)))
        activation x
        [zs activations] (backprop-f net x)]
    (let [delta (mult (cost-derivative (last activations) y) (sigmoid-prime-vec (last zs)))]
      (backprop-b net zs activations delta nabla-b nabla-w))))
    

(defn update-mini-batch [net mini-batch eta]
  (loop [nabla-b (for [b (deref (:biases net))] (matrix 0 (nrow b) (ncol b)))
         nabla-w (for [w (deref (:weights net))] (matrix 0 (nrow w) (ncol w)))
         mb mini-batch
         c (count mini-batch)]
    (if (= c 0)
      (let [z-weights (map vector (deref (:weights net)) nabla-w)
            z-biases (map vector (deref (:biases net)) nabla-b)]
        (let [k (/ eta (count mini-batch))]
        ;(let [k (int (/ eta (count mini-batch)))]
          (reset! (:weights net) (vec (for [[w nw] z-weights] (minus w (mmult k nw)))))
          (reset! (:biases net) (vec (for [[b nb] z-biases] (minus b (mmult k nb)))))))
      (let [[delta-nabla-b delta-nabla-w] (backprop net ((first mb) 0) ((first mb) 1))
            nb (vec (for [[nb dnb] (map vector nabla-b delta-nabla-b)] (plus nb dnb)))
            nw (vec (for [[nw dnw] (map vector nabla-w delta-nabla-w)] (plus nw dnw)))]
        (recur nb
               nw
               (rest mb)
               (dec c))))))
      
(defn SGD [net training-data epochs mini-batch-size eta & [test-data]]
  (loop [j 0]
    (when-not (= j epochs)
      (let [td (shuffle training-data)]
        (let [mini-batches (vec (for [k (range 0 (count training-data) mini-batch-size)]
                                  (vec (take mini-batch-size (drop k td)))))]
;          (for [mini-batch mini-batches]
;            (update-mini-batch net mini-batch eta))))
          (loop [mbs mini-batches]
            (when (first mbs)
              (update-mini-batch net (first mbs) eta)
              (recur (rest mbs))))
      (if test-data
        (println "Epoch " j ": " "evaluate(test-data)" " / " (count test-data))
        (println "Epoch " j " complete"))
      (recur (inc j)))))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))










