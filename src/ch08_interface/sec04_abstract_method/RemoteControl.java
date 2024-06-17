package ch08_interface.sec04_abstract_method;
/*
 * 추상메소드 : public abstract
 */
public interface RemoteControl {
	//상수 필드
	int MAX_VOLUME = 10;
	int MIN_VOLUME = 0;

	//추상 메소드 : public method가 암묵적지정
	void turnOn();
	void turnOff();
	//void setVolume(int volume);
	public abstract void setVolume(int volume);
}