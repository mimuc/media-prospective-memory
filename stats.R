# Copyright 2022 by the author(s) of CHI2023 submission "Short-Form
# Videos Degrade Our Capacity to Retain Intentions: Effect of Context
# Switching On Prospective Memory". All rights reserved.
#
# Use of this source code is governed by a GPLv3 license that
# can be found in the LICENSE file.

library(ARTool)
library(apa)
library(ez)
library(effectsize) # for eta_squared and omega_squared

round_correct <- function(x, digits, chars = TRUE) {
    x <- round(x, digits)
    if(grepl(x = x, pattern = "\\.")) {
        y <- as.character(x)
        pos <- grep(unlist(strsplit(x = y, split = "")), pattern = "\\.", value = FALSE)
        if(chars) {
            return(substr(x = x, start = 1, stop = pos + digits))
        }
        return(
            as.numeric(substr(x = x, start = 1, stop = pos + digits))
        )
    } else {
        return(
            format(round(x, 2), nsmall = 2)
        )
    }
}
twoway_anova_test <- function(df, feature) {
    sig <- shapiro.test(df[[feature]])
    if (sig$p < 0.05) {
        print("not normally distributed:")
        print(sig)
        f <- as.formula(paste(feature, "~ interrupt * measure + (1 | folder_id)"))
        print(f)
        m <- art(f, data = df)
        a <- anova(m)
        print(summary(a))
        print(a, verbose = TRUE)
        print(eta_squared(m, partial = TRUE))
        o <- omega_squared(m)
        print(o)
        print(art.con(m, "interrupt:measure", adjust = "holm") %>%
        summary() %>%
        mutate(sig. = symnum(p.value, corr = FALSE, na = FALSE,
                            cutpoints = c(0, .001, .01, .05, .10, 1),
                            symbols = c("***", "**", "*", ".", " "))))

        # hacking table generalization
        rows <- c(1, 2, 3)
        cols <- c(3, 4, 2, 5)
        line <- c()
        for (r in rows) {
            for (c in cols) {
                if (c == 3 || c == 4) {
                    line <- append(line, as.integer(round_correct(a[r, c], 0)))
                } else {
                    v <- round_correct(a[r, c], 3)
                    if (v < .001) {
                        v <- "\\textbf{< .001}"
                    }
                    if (v < .05) {
                        v <- paste("\\textbf{", v, "}", sep = "")
                    }
                    line <- append(line, v)
                }
            }
            line <- append(line, round_correct(o[r, 2], 3))
        }
        cat(line, sep = " & ")
    } else {
        print("normally distributed:")
        print(sig)
        model <- eval(parse(text = paste0("ezANOVA(data=df, dv=", feature, ", within = .(measure, interrupt), wid = folder_id, detailed = TRUE)")))
        print(summary(model))
        anova_apa(model)

        # hacking table generalization
        a <- model[["ANOVA"]]
        rows <- c(2, 3, 4)
        cols <- c(2, 3, 6, 7, 9)
        line <- c()
        for (r in rows) {
            for (c in cols) {
                if (c == 2 || c == 3) {
                    line <- append(line, as.integer(round_correct(a[r, c], 0)))
                } else {
                    v <- round_correct(a[r, c], 3)
                    if (v < .001) {
                        v <- "\\textbf{< .001}"
                    }
                    if (c == 7 && v < .05) {
                        v <- paste("\\textbf{", v, "}", sep = "")
                    }
                    line <- append(line, v)
                }
            }
        }
        cat(line, sep = " & ")
    }
}