package main

import (
	"fmt"
	"math"
	"strings"
)

func formatSRT(subtitles []Subtitle) string {
	var builder strings.Builder
	for i, subtitle := range subtitles {
		fmt.Fprintf(
			&builder,
			"%d\r\n%s --> %s\r\n%s\r\n\r\n",
			i+1,
			formatSRTTime(subtitle.Start),
			formatSRTTime(subtitle.End),
			strings.TrimSpace(subtitle.Text),
		)
	}
	return builder.String()
}

func formatSRTTime(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	totalMilliseconds := int64(math.Round(seconds * 1000))
	hours := totalMilliseconds / 3_600_000
	minutes := (totalMilliseconds % 3_600_000) / 60_000
	secs := (totalMilliseconds % 60_000) / 1000
	millis := totalMilliseconds % 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
}
