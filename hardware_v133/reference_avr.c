/* ATmega328P 16MHz independent physical capture. D2=photodiode; D3=contact;
 D9=optional isolated actuator, never a timing reference. A,id,delay_us command. */
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/atomic.h>
#include <stdint.h>
#include <stdlib.h>
#define CAP 32
struct Edge{uint32_t trial,ticks;uint8_t kind;};
volatile struct Edge ring[CAP];volatile uint8_t head=0,tail=0,armed=0,actuated=0;volatile uint32_t upper=0,dropped=0,trial=0,delay_ticks=0,onset=0;
static uint32_t stamp(void){uint16_t low=TCNT1;uint32_t hi=upper;if((TIFR1&(1<<TOV1))&&low<32768)hi+=65536UL;return hi+low;}
static void push(uint8_t k,uint32_t t){uint8_t next=(head+1)%CAP;if(next==tail){dropped++;return;}ring[head].trial=trial;ring[head].ticks=t;ring[head].kind=k;head=next;}
ISR(TIMER1_OVF_vect){upper+=65536UL;}
ISR(INT0_vect){uint32_t at=stamp();if(armed==1){onset=at;armed=2;actuated=0;push('S',at);}else push('X',at);}
ISR(INT1_vect){uint32_t at=stamp();if(armed==2){push('R',at);armed=0;PORTB&=~(1<<PB1);}else push('U',at);}
static void tx(char c){while(!(UCSR0A&(1<<UDRE0)));UDR0=c;}
static void str(const char*s){while(*s)tx(*s++);}
static void num(uint32_t n){char b[11];ultoa(n,b,10);str(b);}
static void command(char*s){if(s[0]!='A'||s[1]!=',')return;char*end;uint32_t id=strtoul(s+2,&end,10);if(*end!=',')return;uint32_t delay=strtoul(end+1,&end,10);if(*end||delay>10000000UL)return;ATOMIC_BLOCK(ATOMIC_RESTORESTATE){trial=id;delay_ticks=delay*2UL;armed=1;actuated=0;PORTB&=~(1<<PB1);}str("ARMED,");num(id);tx('\n');}
int main(void){DDRD&=~((1<<PD2)|(1<<PD3));DDRB|=1<<PB1;TCCR1A=0;TCCR1B=1<<CS11;TIMSK1=1<<TOIE1;EICRA=(1<<ISC01)|(1<<ISC00)|(1<<ISC11)|(1<<ISC10);EIMSK=(1<<INT0)|(1<<INT1);UBRR0H=0;UBRR0L=8;UCSR0B=(1<<RXEN0)|(1<<TXEN0);UCSR0C=(1<<UCSZ01)|(1<<UCSZ00);sei();str("trial,edge,ref_ticks,clock_id,dropped\n");char line[48];uint8_t n=0;
 for(;;){if(UCSR0A&(1<<RXC0)){char c=UDR0;if(c=='\n'){line[n]=0;command(line);n=0;}else if(c!='\r'){if(n<sizeof(line)-1)line[n++]=c;else n=0;}}
 ATOMIC_BLOCK(ATOMIC_RESTORESTATE){if(armed==2&&!actuated&&(uint32_t)(stamp()-onset)>=delay_ticks){PORTB|=1<<PB1;actuated=1;}if(armed==2&&actuated&&(uint32_t)(stamp()-onset)>=delay_ticks+100000UL){PORTB&=~(1<<PB1);}if(armed==2&&(uint32_t)(stamp()-onset)>20000000UL){push('T',stamp());armed=0;PORTB&=~(1<<PB1);}}
 if(tail!=head){struct Edge e;uint32_t lost;ATOMIC_BLOCK(ATOMIC_RESTORESTATE){e=ring[tail];tail=(tail+1)%CAP;lost=dropped;}num(e.trial);tx(',');tx(e.kind);tx(',');num(e.ticks);str(",ATMEGA328P_TIMER1,");num(lost);tx('\n');}}
}
